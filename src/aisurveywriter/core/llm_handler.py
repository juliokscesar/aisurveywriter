from enum import Enum, auto
from typing import Optional, Union
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, AIMessage
from langchain.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

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
                self.llm = ChatOpenAI(model=model, temperature=0.3)
            case LLMType.Google:
                self.llm = ChatGoogleGenerativeAI(model=model, temperature=0.3)
            case _:
                raise ValueError(f"Invalid model type: {model_type}")

        self.prompt = None
        self._chain = None
    
    def init_chain(self, ctxmsg: SystemMessage, prompt: str):
        input_prompt = ChatPromptTemplate.from_messages([
            ctxmsg, 
            HumanMessagePromptTemplate.from_template(prompt),
        ])

        self.prompt = input_prompt
        self._chain = input_prompt | self.llm
    
    def set_prompt_template(self, prompt: str):
        self.prompt = prompt

    def invoke(self, input_variables: dict) -> AIMessage:
        """
        Invokes langchain LLM object and changes all the "input_variables" in the prompt template.
        Must have called init_chain, set_prompt_template and (opt.) set_context_sysmsg before
        """
        if self._chain is None:
            raise RuntimeError("To call LLMHandler.invoke, the chain has to be initialized")
        return self._chain.invoke(input_variables)

    def send_prompt(self, prompt: str) -> AIMessage:
        return self.llm.invoke(prompt)
