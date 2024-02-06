from langchain.llms import LLM  # This is the base class for LLMs in LangChain
from langchain.openai import OpenAI  # Predefined wrapper for OpenAI
from langchain.huggingface import HuggingFace  # Predefined wrapper for Hugging Face models

# Custom LLM Wrapper that can leverage different providers
class CustomLLMWrapper:
    def __init__(self, provider, model_name, **kwargs):
        if provider == "openai":
            self.llm = OpenAI(model_name=model_name, **kwargs)
        elif provider == "huggingface":
            self.llm = HuggingFace(model_name=model_name, **kwargs)
        # Add more elif blocks here for other providers
        else:
            raise ValueError(f"Provider {provider} not supported.")

    def invoke(self, prompt, **kwargs):
        return self.llm.invoke(prompt, **kwargs)

    def stream(self, prompt, **kwargs):
        if hasattr(self.llm, "stream"):
            return self.llm.stream(prompt, **kwargs)
        else:
            raise NotImplementedError(f"Streaming not implemented for {type(self.llm).__name__}")
