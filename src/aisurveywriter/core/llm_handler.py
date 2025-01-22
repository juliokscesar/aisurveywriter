from enum import Enum, auto
from typing import Union
from langchain_openai import ChatOpenAI
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.messages import SystemMessage, AIMessage
from langchain.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

class LLMType(Enum):
    OpenAI = auto()
    Google = auto()

    @staticmethod
    def from_str(s: str):
        match s.strip().lower():
            case "openai":
                return LLMType.OpenAI
            case "google":
                return LLMType.Google
            case _:
                raise ValueError(f"{s!r} is not a valid LLMType")

class LLMHandler:
    def __init__(self, model: str, model_type: Union[LLMType, str]):
        self.model = model
        if isinstance(model_type, str):
            model_type = LLMType.from_str(model_type)
        match model_type:
            case LLMType.OpenAI:
                self.llm = ChatOpenAI(model=model)
            case LLMType.Google:
                self.llm = GoogleGenerativeAI(model=model)
            case _:
                raise ValueError(f"Invalid model type: {model_type}")

        self.prompt = None
        self._chain = None
        self._ctx: SystemMessage = None
    
    def init_chain(self, ctxmsg: SystemMessage, prompt: str):
        input_prompt = HumanMessagePromptTemplate.from_template(prompt)
        input_prompt = ChatPromptTemplate.from_messages([ctxmsg, input_prompt])

        self.prompt = input_prompt
        self._chain = input_prompt | self.llm
    
    def set_context_sysmsg(self, ctx: SystemMessage):
        self._ctx = ctx

    def set_prompt_template(self, prompt: str):
        self.prompt = prompt

    def write(self, input_variables: dict) -> AIMessage:
        """
        Invokes langchain LLM object and changes all the "input_variables" in the prompt template.
        Must have called init_chain, set_prompt_template and (opt.) set_context_sysmsg before
        """
        return self._chain.invoke(input_variables)
