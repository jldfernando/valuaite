import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_mistralai import ChatMistralAI
from langchain_openai import ChatOpenAI

def get_llm(provider: str = "Gemini", model_name: str = None, api_key: str = None):
    """
    Factory function to get the appropriate LLM based on provider.
    """
    if provider == "Gemini":
        return ChatGoogleGenerativeAI(
            model=model_name or "gemini-2.5-flash-lite", 
            google_api_key=api_key or os.getenv("GOOGLE_API_KEY"),
            max_retries=0
        )
    elif provider == "Groq":
        return ChatGroq(
            model=model_name or "llama-3.3-70b-versatile",
            api_key=api_key or os.getenv("GROQ_API_KEY"),
            max_retries=0
        )
    elif provider == "Mistral":
        return ChatMistralAI(
            model=model_name or "mistral-large-latest",
            api_key=api_key or os.getenv("MISTRAL_API_KEY"),
            max_retries=0
        )
    elif provider == "OpenAI":
        return ChatOpenAI(
            model=model_name or "gpt-4o-mini",
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            max_retries=0
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")
