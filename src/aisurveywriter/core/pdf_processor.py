from typing import List, Union, Tuple
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
import fitz
from pathlib import Path
import re
from pydantic import BaseModel
from PIL import Image
import io

from aisurveywriter.utils import named_log

class PDFImageData(BaseModel):
    pdf_id: int
    page_id: int
    page_image_id: int
    data: bytes
    ext: str
    path: str
    caption: Union[str, None] = None

class PDFProcessor:
    def __init__(self, pdf_paths: List[str]):
        self.pdf_paths: List[str] = pdf_paths
        self.pdf_documents: List[List[Document]] = [None] * len(pdf_paths)
        self.img_readers: List[fitz.Document] = [None] * len(pdf_paths)
        self._load()
        

    def _load(self):
        for i, pdf in enumerate(self.pdf_paths):
            self.pdf_documents[i] = PyPDFLoader(pdf).load()
            self.img_readers[i] = fitz.open(pdf)

    def _print(self, *msgs: str):
        print(f"({self.__class__.__name__})", *msgs)

    def extract_content(self) -> List[str]:
        contents = [""] * len(self.pdf_documents)
        for i, doc in enumerate(self.pdf_documents):
            contents[i] = "\n".join([d.page_content for d in doc])
        return contents

    def extract_images(self, save_dir: str = None, verbose=True, filter_min_wh: Tuple[int,int] = None, extract_captions=True) -> List[PDFImageData]:
        imgs = []
        if save_dir:
            save_dir = os.path.abspath(save_dir)
            os.makedirs(save_dir, exist_ok=True)
            
        for pdf_idx, reader in enumerate(self.img_readers):
            for page_idx in range(len(reader)):
                page = reader.load_page(page_idx)
                img_list = page.get_images(full=True)
                
                # Temporary storage for images with their positions
                page_images = []
                
                page_width = page.rect.width
                column_split = page_width / 2
                
                for img_idx, img in enumerate(img_list):
                    xref = img[0]
                    
                    # Get the image's position on the page
                    img_rect = page.get_image_bbox(img)
                    
                    # Extract the image data
                    base_img = reader.extract_image(xref)
                    img_bytes = base_img["image"]
                    img_ext = base_img["ext"]
                    
                    # Apply size filter if specified
                    if filter_min_wh:
                        img_wh = Image.open(io.BytesIO(img_bytes)).size
                        if img_wh[0] < filter_min_wh[0] or img_wh[1] < filter_min_wh[1]:
                            continue
                    
                    # Determine which column the image belongs to based on its center x-coordinate
                    img_center_x = (img_rect.x0 + img_rect.x1) / 2
                    column = 0 if img_center_x < column_split else 1  # 0 = left column, 1 = right column
                    
                    # Store the image with its position, column, and data
                    page_images.append({
                        'column': column,          # Column (left=0, right=1)
                        'y_pos': img_rect.y0,      # Vertical position
                        'x_pos': img_rect.x0,      # Horizontal position
                        'rect': img_rect,          # Full bounding box
                        'orig_idx': img_idx,       # Original index
                        'xref': xref,              # XRef for the image
                        'bytes': img_bytes,        # Image binary data
                        'ext': img_ext             # Image extension
                    })
                
                # Sort images: first by column (left to right), then by vertical position (top to bottom)
                page_images.sort(key=lambda x: (x['column'], x['y_pos']))
                
                # Process the sorted images
                for sorted_idx, img_info in enumerate(page_images):
                    img_name = f"{Path(self.pdf_paths[pdf_idx]).stem}_page{page_idx}_image{sorted_idx}.{img_info['ext']}"
                    
                    if save_dir:
                        path = os.path.join(save_dir, img_name)
                        with open(path, "wb") as f:
                            f.write(img_info['bytes'])
                        
                        if verbose:
                            column_name = "left" if img_info['column'] == 0 else "right"
                            named_log(self, f"Image saved: {path} (from {column_name} column, y-pos: {img_info['y_pos']:.1f})")

                    imgs.append(PDFImageData(
                        pdf_id=pdf_idx,
                        page_id=page_idx,
                        page_image_id=sorted_idx,  # Now this is the column-aware, top-to-bottom index
                        data=img_info['bytes'],
                        ext=img_info['ext'],
                        path=os.path.join(save_dir, img_name) if save_dir else img_name
                    ))
                    
        if extract_captions:
            imgs = self.extract_image_captions(imgs)
        
        return imgs
            
    def extract_image_captions(self, images_data: List[PDFImageData]) -> List[PDFImageData]:
        captions = []
        # caption_pattern = re.compile(r"^(Figure|Fig\.|Figura)\s*(\d+)(\s*\.(\s*)|\s+)(\(\w+\)\s*|\d+\s*)?(?=[A-Z])")
        caption_pattern = re.compile(r"^(Figure|Fig\.|Figura)\s*(?:\n\s*)*(\d+)\.?\s*", re.MULTILINE)
    
        for pdf_idx, pdf_doc in enumerate(self.pdf_documents):
            for page_idx, page in enumerate(pdf_doc):
                text = page.page_content.strip()
                if not text: # skip empty pages
                    continue

                # fix figure captions that are loaded with linebreaks between "Figure" and the number
                text = re.sub(caption_pattern, r"\1 \2 ", text)

                lines = text.split("\n")
                page_caption_count = 0
                current_caption_text = None

                for line_idx, line in enumerate(lines):
                    if caption_match := caption_pattern.search(line):
                        # skip pattern occurrences within text (i.e. figure citation instead of caption)
                        caption_words = line[caption_match.end():].strip().split()
                        if not caption_words or len(caption_words) < 4:
                            continue
                        
                        # possible citations: 
                        # - first word after number is lowercase and it's not within parenthesis
                        first_word = caption_words[0]
                        if not first_word.startswith('('):
                            if first_word.strip()[0] in ['.', ',', ';', '!', '?', ')', '/', '\\', '[', ']', '{', '}']:
                                continue
                            if first_word.islower() or (len(first_word) == 1 and caption_words[1].strip().lower() == "shows"):
                                continue
                        
                        if current_caption_text:
                            captions.append((pdf_idx, page_caption_count, page_idx, current_caption_text.strip()))
                            page_caption_count += 1
                        
                        current_caption_text = line.strip()
                    
                    elif current_caption_text:
                        current_caption_text += " " + line.strip()
                        
                        if line_idx < len(lines) - 1 and lines[line_idx + 1].strip() == "":
                            captions.append((pdf_idx, page_caption_count, page_idx, current_caption_text.strip()))
                            page_caption_count += 1
                            current_caption_text = None

                if current_caption_text:
                    captions.append((pdf_idx, page_caption_count, page_idx, current_caption_text.strip()))                        

        # match captions with images based on page number
        matched_ids = set()
        for img in images_data:
            for caption_idx, (pdf_caption_id, page_caption_id, caption_page, caption_text) in enumerate(captions):
                if caption_idx in matched_ids: 
                    continue
                
                if img.page_id == caption_page and page_caption_id == img.page_image_id:
                    matched_ids.add(caption_idx)
                    img.caption = caption_text.strip()
                    break

        return images_data

    def summarize_content(self, summarizer, chunk_size: int = 2000, chunk_overlap: int = 200, show_metadata = False) -> List[str]:
        # Split all pdf content in smaller chunks
        text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = text_splitter.split_documents(self.pdf_documents)

        # Summarize each chunk
        size = len(chunks)
        summaries = [None] * size
        for i, chunk in enumerate(chunks):
            summary = summarizer.invoke(f"Summarize the content of the following text:\n\n{chunk}")
            if show_metadata:
                self._print(f"Summary response metadata (chunk {i+1}/{size}):", summary.usage_metadata)
            summaries[i] = summary.content
        
        return "\n".join(summaries)
