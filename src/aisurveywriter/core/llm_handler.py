from enum import Enum, auto
from langchain_openai import ChatOpenAI
from langchain_google_genai import GoogleGenerativeAI

class LLMType(Enum):
    OpenAI = auto()
    Google = auto()

class LLMHandler:
    def __init__(self, model: str, model_type: LLMType):
        self.model = model
        match model_type:
            case LLMType.OpenAI:
                self.llm = ChatOpenAI(model=model)
            case LLMType.Google:
                self.llm = GoogleGenerativeAI(model=model)
            case _:
                raise ValueError(f"Invalid model type: {model_type}")
    
    