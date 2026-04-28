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
    Initialize a LangChain chat model. Only the OpenAI-compatible provider is supported;
    self-hosted backends such as vLLM or Ollama should be accessed via their OpenAI-compatible
    endpoints using llm='openai' with the corresponding base_url.

    :param llm: provider name, must be 'openai'.
    :param model_name: model name, e.g. 'gpt-4o-mini'.
    :param role: 'index' | 'rag' | None. Selects which env vars provide api_key/base_url.
    """
    if llm != 'openai':
        raise NotImplementedError(
            f"LLM provider '{llm}' is not supported. Use 'openai' (works for OpenAI and any "
            "OpenAI-compatible endpoint such as vLLM or Ollama's /v1 API)."
        )
    from langchain_openai import ChatOpenAI
    api_key, base_url = _resolve_openai_credentials(role)
    return ChatOpenAI(api_key=api_key, base_url=base_url, model=model_name, temperature=temperature, max_retries=max_retries, timeout=timeout, seed=seed, **kwargs)


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
