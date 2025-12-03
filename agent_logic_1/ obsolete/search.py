# Search
# from dotenv import load_dotenv
# from langchain_community.tools.tavily_search import TavilySearchResults
from tavily import TavilyClient
import time

# os.environ["TAVILY_API_KEY"] = "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"

# _ = load_dotenv()


def web_search(question: str, outputs_number: int = 2):
    """
        Не работает так же как синхронная функция.
    Есть предположение, что выдает строку вместо списка словарей,
    где каждый словарь имеет ключ 'content'
    """
    tavily_client = TavilyClient()
    start_time = time.time()
    context = tavily_client.get_search_context(query=question, max_results=outputs_number, max_tokens=1000)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(
        f"Tavily research timing: {elapsed_time:.2f} sec")
    return context
