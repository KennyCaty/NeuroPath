import argparse
import os

import tiktoken

enc = tiktoken.encoding_for_model("gpt-3.5-turbo")


def num_tokens_by_tiktoken(text: str):
    return len(enc.encode(text))


class LangChainModel:
    def __init__(self, provider: str, model_name: str, **kwargs):
        self.provider = provider
        self.model_name = model_name
        self.kwargs = kwargs


def _resolve_openai_credentials(role: str | None):
    """Return (api_key, base_url) based on role.

    role='index' -> INDEX_LLM_API_KEY / INDEX_LLM_BASE_URL (fallback to OPENAI_*)
    role='rag'   -> RAG_LLM_API_KEY   / RAG_LLM_BASE_URL   (fallback to OPENAI_*)
    role=None    -> OPENAI_API_KEY / OPENAI_BASE_URL
    """
    default_key = os.environ.get("OPENAI_API_KEY")
    default_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    if role == "index":
        return (os.environ.get("INDEX_LLM_API_KEY", default_key),
                os.environ.get("INDEX_LLM_BASE_URL", default_url))
    if role == "rag":
        return (os.environ.get("RAG_LLM_API_KEY", default_key),
                os.environ.get("RAG_LLM_BASE_URL", default_url))
    return default_key, default_url


def init_langchain_model(llm: str, model_name: str = "gpt-4o-mini", temperature: float = 0.0, max_retries=5, timeout=60, seed=42, role: str | None = None, **kwargs):
    """
    Initialize a language model from the langchain library.
    :param llm: The LLM to use, e.g., 'openai', 'together'
    :param model_name: The model name to use, e.g., 'gpt-3.5-turbo'
    :param role: 'index' | 'rag' | None. Selects which env vars provide api_key/base_url for OpenAI-compatible endpoints.
    """
    if llm == 'openai':
        from langchain_openai import ChatOpenAI
        api_key, base_url = _resolve_openai_credentials(role)
        return ChatOpenAI(api_key=api_key, base_url=base_url, model=model_name, temperature=temperature, max_retries=max_retries, timeout=timeout, seed=seed, **kwargs)
    elif llm == 'together':
        from langchain_together import ChatTogether
        return ChatTogether(api_key=os.environ.get("TOGETHER_API_KEY"), model=model_name, temperature=temperature, **kwargs)
    elif llm == 'ollama':
        from langchain_ollama import ChatOllama
        return ChatOllama(model=model_name)
    elif llm == 'llama.cpp':
        from langchain_community.chat_models import ChatLlamaCpp
        return ChatLlamaCpp(model_path=model_name, verbose=True)
    else:
        raise NotImplementedError(f"LLM '{llm}' not implemented yet.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--llm', type=str)
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--query', type=str, help='query text', default="who are you?")
    args = parser.parse_args()

    model = init_langchain_model(args.llm, args.model_name)
    messages = [("system", "You are a helpful assistant. Please answer the question from the user."), ("human", args.query)]
    completion = model.invoke(messages)
    print(completion.content)
