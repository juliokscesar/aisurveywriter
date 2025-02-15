from enum import Enum, auto
from typing import Union, List

from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM
import torch

class HighMemoryEmbeddings(Embeddings):
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", **model_kwargs):
        """
        Custom LangChain embedding class that loads a Hugging Face model
        using 8-bit quantization via bitsandbytes.
        
        :param model_name: Name of the Hugging Face model to load.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._quantization_cfg = BitsAndBytesConfig(load_in_8bit=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=self._quantization_cfg,
            device_map="auto",  # Automatically assign to available device
            **model_kwargs,
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of documents.
        
        :param texts: List of text strings to embed.
        :return: List of embedding vectors.
        """
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).cpu().tolist()  # Mean pooling
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query text.
        
        :param text: Input text string.
        :return: Embedding vector.
        """
        return self.embed_documents([text])[0]


class EmbedModelType(Enum):
    OpenAI      = auto()
    Google      = auto()
    HuggingFace = auto()
    HighMemory  = auto()
    
    @staticmethod
    def from_str(s: str):
        match s.strip().lower():
            case "openai":
                return EmbedModelType.OpenAI
            case "google":
                return EmbedModelType.Google
            case "highmemory":
                return EmbedModelType.HighMemory
            case "huggingface":
                return EmbedModelType.HuggingFace
            case _:
                return EmbedModelType.HuggingFace

def load_embeddings(model: str, model_type: Union[EmbedModelType, str], **model_kwargs):
    if isinstance(model_type, str):
        model_type = EmbedModelType.from_str(model_type)
    match model_type:
        case EmbedModelType.OpenAI:
            return OpenAIEmbeddings(model=model, **model_kwargs)
        case EmbedModelType.Google:
            return GoogleGenerativeAIEmbeddings(model=model, **model_kwargs)
        case EmbedModelType.HuggingFace:
            return HuggingFaceEmbeddings(model_name=model, model_kwargs=model_kwargs)
        case EmbedModelType.HighMemory:
            return HighMemoryEmbeddings(model_name=model, **model_kwargs)
        case _:
            raise ValueError("Invalid model type:", model_type)
