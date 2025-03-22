from typing import List, Optional
import layoutparser as lp
import pdf2image
import numpy as np
import cv2
from PIL import Image
import re
import os
from scipy.spatial import KDTree

from .document import Document, DocPage, DocFigure
from .lp_handler import LayoutParserAgents, LayoutParserSettings
from ..utils.logger import named_log
from ..utils.helpers import get_bibtex_entry

def sort_blocks_article_layout(blocks, page_width, page_height) -> lp.Layout:
    """
    Sort blocks for article layout, that is:
    Top-Left Blocks -> Bottom-Left Blocks -> Top-Right Blocks -> Bottom-Right blocks
    """
    # Define page midpoints
    mid_x = page_width / 2
    mid_y = page_height / 2
    
    # Group blocks by quadrant
    top_left = []
    top_right = []
    bottom_left = []
    bottom_right = []
    
    for block in blocks:
        # Get block center
        x1, y1, x2, y2 = block.coordinates
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        # Assign to quadrant
        if center_x < mid_x:
            if center_y < mid_y:
                top_left.append(block)
            else:
                bottom_left.append(block)
        else:
            if center_y < mid_y:
                top_right.append(block)
            else:
                bottom_right.append(block)
    
    # Sort blocks within each quadrant by y-coordinate (top to bottom)
    for quadrant in [top_left, top_right, bottom_left, bottom_right]:
        quadrant.sort(key=lambda block: block.coordinates[1])
    
    # Combine quadrants in reading order: top-left, bottom-left, top-right, bottom-right
    return lp.Layout(top_left + bottom_left + top_right + bottom_right)

def blocks_iou(block1, block2):
    # Extract coordinates
    x1_1, y1_1, x2_1, y2_1 = block1.coordinates
    x1_2, y1_2, x2_2, y2_2 = block2.coordinates
    
    # Calculate intersection area
    x_left = max(x1_1, x1_2)
    y_top = max(y1_1, y1_2)
    x_right = min(x2_1, x2_2)
    y_bottom = min(y2_1, y2_2)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
        
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    # Calculate box areas
    block1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    block2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    
    # Calculate IoU
    iou = intersection_area / float(block1_area + block2_area - intersection_area)
    return iou

def layout_nms(blocks, iou_threshold: float = 0.8, neighbors_radius: float = 1.5) -> lp.Layout:
    """
    Apply Non-Maximum Supression to remove duplicate blocks by comparing the IOU
    to "iou_threshold" between the "anchor" block and all neighbors over a radius 
    of "neighbors_radius", using KDTree for fast-neighbor search.
    """
    if not blocks:
        return blocks
    
    # create kdtree based on centers
    centers = np.array([[(b.coordinates[0] + b.coordinates[2]) / 2, 
                        (b.coordinates[1] + b.coordinates[3]) / 2] for b in blocks])
    blocks_kdtree = KDTree(centers)
    
    # search with radius based on average size
    avg_width = np.mean([b.coordinates[2] - b.coordinates[0] for b in blocks])
    avg_height = np.mean([b.coordinates[3] - b.coordinates[1] for b in blocks])
    search_radius = max(avg_width, avg_height) * neighbors_radius

    blocks_to_remove = set()
    for i, block in enumerate(blocks):
        if i in blocks_to_remove:
            continue
        
        neighbor_indices = blocks_kdtree.query_ball_point(centers[i], search_radius)
        for j in neighbor_indices:
            if i == j or j in blocks_to_remove or block.type != blocks[j].type:
                continue
        
            if blocks_iou(blocks[i], blocks[j]) < iou_threshold:
                continue
            
            # keep the one block which has higher area
            area_i = (block.coordinates[2] - block.coordinates[0]) * (block.coordinates[3] - block.coordinates[1])
            area_j = (blocks[j].coordinates[2] - blocks[j].coordinates[0]) * (blocks[j].coordinates[3] - blocks[j].coordinates[1])
            if area_i < area_j:
                blocks_to_remove.add(i)
                break
            else:
                blocks_to_remove.add(j)
                
    filtered_blocks = [b for i, b in enumerate(blocks) if i not in blocks_to_remove]
    return lp.Layout(filtered_blocks)
        


class PDFProcessor:
    def __init__(self, pdf_paths: List[str],
                 lp_settings: LayoutParserSettings,
                 images_output_dir: str = "output",
                 parse_threads: int = 3):
        # check and use existing pdf paths
        self.pdf_paths: List[str] = []
        for path in pdf_paths:
            if not os.path.isfile(path):
                named_log(self, "skipping file not found:", path)
            else:
                self.pdf_paths.append(path)
        
        # create layout parser agents
        self.lp_agents = LayoutParserAgents(lp_settings.config_path, lp_settings.score_threshold,
                                             lp_settings.tesseract_executable)
        
        self.images_output_dir = images_output_dir
        os.makedirs(self.images_output_dir, exist_ok=True)
        
        # parse pdfs to Documents
        self.documents: List[Document] = []

        self._parser_threads = parse_threads
        self._caption_pattern = re.compile(r"^(fig|figure|figura|scheme)(?:.+?)(\d+)", re.IGNORECASE)
        self.parse_pdfs()
        
    def parse_pdfs(self, reload=False) -> List[Document]:
        """
        Parse all PDFs using LayoutParser, building a Document object
        for every file. All images gathered are save in the directory
        self.images_output_dir
        
        This is called within the constructor. If "reload" is provided,
        it will parse all PDFs again, discarding those parsed in the constructor.
        """
        assert(len(self.pdf_paths) >= 1)
        
        if self.documents and not reload:
            return self.documents
        
        pdf_amount = len(self.pdf_paths)
        for pdf_i, pdf_path in enumerate(self.pdf_paths):
            print()
            named_log(self, f"started processing PDF {pdf_i+1}/{pdf_amount}:", os.path.basename(pdf_path))
            
            page_images = pdf2image.convert_from_path(pdf_path, thread_count=self._parser_threads)
            page_images = [img.convert("RGB") for img in page_images]
            page_amount = len(page_images)

            doc_pages: List[DocPage] = []
            doc_figures: List[DocFigure] = []            
            doc_title: str = None
            doc_authors: str = None
            title_layout: lp.Layout = None
            for page_num, page_image in enumerate(page_images):

                # look for title and author if on first page
                if page_num == 0:
                    doc_page, page_figures, title_layout = self._parse_page_image(page_image, pdf_path, page_num, return_layout=True)
                    title_blocks = [b for b in title_layout if b.type == "Title"]
                    if title_blocks:
                        doc_title = title_blocks[0].text
                        # assume our authors names are on text block closest and below to the title
                        min_distance = float("inf")
                        authors_block = None
                        title_center = np.array([(title_blocks[0].coordinates[0] + title_blocks[0].coordinates[2]) / 2, (title_blocks[0].coordinates[1] + title_blocks[0].coordinates[3]) / 2])
                        for block in title_layout:
                            if block.type != "Text":
                                continue
                            block_center = np.array([(block.coordinates[0] + block.coordinates[2]) / 2, (block.coordinates[1] + block.coordinates[3]) / 2])
                            if block_center[1] < title_center[1]:
                                continue
                            
                            distance = np.linalg.norm(title_center - block_center)
                            if distance < min_distance:
                                authors_block = block
                                break
                        
                        if authors_block:
                            doc_authors = authors_block.text
                            
                else:    
                    doc_page, page_figures = self._parse_page_image(page_image, pdf_path, page_num)

                doc_pages.append(doc_page)
                doc_figures.extend(page_figures)
                named_log(self, f"processed page {page_num+1}/{page_amount}, figures with caption extracted: {len([fig for fig in page_figures if fig.caption])}")
            
            # try to get bibtex entry
            try:
                bibtex_entry = get_bibtex_entry(doc_title, None)
                if not bibtex_entry:
                    named_log(self, "unable to get bibtex entry for", pdf_path)
            except Exception as e:
                named_log(self, "exception raised when getting pdf bibtex entry:", e)
                bibtex_entry = None        
            
            if bibtex_entry:
                pdf_basename = os.path.basename(pdf_path).removesuffix(".pdf")
                # use authors from bibtex entry if found
                doc_authors = bibtex_entry.get("author", doc_authors)
                bibtex_entry["ID"] = f"key_pdf{pdf_basename}"
            
            doc = Document(
                path=pdf_path, 
                title=doc_title,
                author=doc_authors,
                bibtex_entry=bibtex_entry,
                pages=doc_pages,
                figures=doc_figures,
            )
            self.documents.append(doc)

        return self.documents

    def _parse_page_image(self, page_image: Image.Image, source_pdf: str, page_num: int, return_layout=False) -> tuple[DocPage, List[DocFigure], Optional[lp.Layout]]:
        img_np = np.array(page_image)
        page_width, page_height = page_image.width, page_image.height
        source_basename = os.path.basename(source_pdf)
        source_basename = source_basename[:source_basename.rfind(".pdf")]
        
        # parse layout blocks and sort them
        layout = self.lp_agents.model.detect(img_np)
        layout = sort_blocks_article_layout(layout, page_width, page_height)

                
        # separate between text and figure blocks
        text_blocks = lp.Layout([b for b in layout if b.type in ["Text", "Title", "List"]])
        figure_blocks = lp.Layout([b for b in layout if b.type == "Figure"])
        
        # remove text detected within figures by comparing iou
        text_blocks = lp.Layout([b for b in text_blocks if not any(blocks_iou(b, b_fig)>0.8 for b_fig in figure_blocks)])
        
        layout = text_blocks + figure_blocks
        
        # remove duplicates with NMS
        layout = layout_nms(layout)

        # extract text from text blocks using OCR
        page_text: List[str] = []
        for block in text_blocks:
            # crop text block from page image
            segment_image = block.pad(left=5,right=5,top=5,bottom=5).crop_image(img_np)
        
            # extract text with ocr
            text = self.lp_agents.ocr.detect(segment_image)
            block.text = text
            page_text.append(text)

        # process figures and associate captions
        page_figures: List[DocFigure] = []
        for i, figure_block in enumerate(figure_blocks):
            figure_img = figure_block.crop_image(img_np)
            fig_x1, fig_y1, fig_x2, fig_y2 = [int(coord) for coord in figure_block.coordinates]

            # set caption to be the text closest to the figure
            caption = ""
            fig_id = -1
            min_distance = float("inf")
            for text_block in text_blocks:
                text_y1 = text_block.coordinates[1]
                text_x_center = (text_block.coordinates[0] + text_block.coordinates[2]) / 2
                fig_x_center = (fig_x1 + fig_x2) / 2
                
                # check if text block is below the figure and horizontally aligned
                if (text_y1 > fig_y2 and 
                    abs(text_x_center - fig_x_center) < (fig_x2 - fig_x1) / 2):
                    
                    distance = text_y1 - fig_y2
                    if distance < min_distance:
                        min_distance = distance
                        caption = text_block.text
                        
                        # caption threshold - if it's too far, probably not a caption
                        # also check caption pattern (look for figure,fig, or some preffix that indicates that this is a caption)
                        caption_match = self._caption_pattern.match(caption)
                        if min_distance > 100 or not caption_match:
                            caption = None
                        else:
                            fig_id = int(caption_match.group(2))
            
            # skip figure if no caption found
            if not caption:
                continue
            
            # clean caption text
            if caption:
                caption = caption.strip()
                # remove extra newlines
                caption = re.sub(r"\n+", " ", caption)
            
            # save figure
            figure_filename = f"{source_basename}_page{page_num}_image{i}.png"
            figure_path = os.path.join(self.images_output_dir, figure_filename)
            Image.fromarray(figure_img).save(figure_path)
            
            doc_figure = DocFigure(id=fig_id, image_path=figure_path, caption=caption, source_path=source_pdf)
            page_figures.append(doc_figure)

        # parse to DocPage object
        page = DocPage(id=page_num, content="\n".join(page_text), source_path=source_pdf)

        if return_layout:
            return page, page_figures, text_blocks + figure_blocks
        else:
            return page, page_figures
    