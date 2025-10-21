# Важно! При продакшене необходимо включить CUDA!
from langchain_core.documents import Document
from transformers import AutoTokenizer, AutoModel
import torch

# Загрузка предобученной модели и токенизатора
# TODO: Решить вопрос с кешированием модели, чтобы она не загружалась с huggingface.co каждый раз.
# cache_dir="/app/cache"
# По умолчанию выдает ошибку, нужно городить эту системную папку в контейнере и делать монтирование в хостовую систему.
tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny2", )
model = AutoModel.from_pretrained("cointegrated/rubert-tiny2")


#
# model.cuda()  # uncomment it if GPU
#

# Определение запроса и ключевого слова
# query = "Ваш поисковый запрос"
# keyword = "ключевое слово"

# Выполнение поиска
# results = ...  # retriever.get_relevant_documents(query)


# Функция для получения эмбеддинга текста
def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)


def filtrate(keyword: str, texts: list[Document], threshold: float = 0.005) -> list[Document]:
    """
    Фильтрует документы на основе косинусного сходства с заданным ключевым словом,
    используя динамически устанавливаемое пороговое значение.
    """
    # Эмбеддинг ключевого слова
    keyword_embedding = get_embedding(keyword)
    print("- Section FILTRATE -")
    print("keyword: ", keyword)
    print("Keyword Shape: ", keyword_embedding.shape)

    # Вычисление сходства и определение максимального значения
    similarities = []
    if texts is not None:
        for txt in texts:
            doc_embedding = get_embedding(txt.page_content)
            similarity = torch.nn.functional.cosine_similarity(keyword_embedding, doc_embedding)
            similarities.append((txt, similarity.item()))
            print("similarity.item(): ", similarity.item())
    else:
        print("Document for FILTRATION was not given, maybe because of connection error")
        return []

    # Определение порогового значения
    max_similarity = max(similarities, key=lambda x: x[1])[1]
    dynamic_threshold = max_similarity - threshold
    print("Dynamic Threshold: ", dynamic_threshold)

    # Фильтрация документов по пороговому значению
    filtered_results = [doc for doc, sim in similarities if sim >= dynamic_threshold]

    return filtered_results
