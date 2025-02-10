from enum import Enum, auto
from typing import Union

from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

class EmbedModelType(Enum):
    OpenAI      = auto()
    Google      = auto()
    HuggingFace = auto()
    
    @staticmethod
    def from_str(s: str):
        match s.strip().lower():
            case "openai":
                return EmbedModelType.OpenAI
            case "google":
                return EmbedModelType.Google
            case _:
                return EmbedModelType.HuggingFace

def load_embeddings(model: str, model_type: Union[EmbedModelType, str]):
    if isinstance(model_type, str):
        model_type = EmbedModelType.from_str(model_type)
    match model_type:
        case EmbedModelType.OpenAI:
            return OpenAIEmbeddings(model=model)
        case EmbedModelType.Google:
            return GoogleGenerativeAIEmbeddings(model=model)
        case EmbedModelType.HuggingFace:
            return HuggingFaceEmbeddings(model_name=model)
        case _:
            raise ValueError("Invalid model type:", model_type)