import os
import pandas as pd
from scr.searcher import Searcher
from scr.texts_processing import TokensVectors, TextShingles

shinglorizer = TextShingles(3)
vectorizer = TokensVectors(3010)
searcher = Searcher()


if __name__ == "__main__":
    """1) Добавление данных в сервис:"""
    df = pd.read_csv(os.path.join("data", "queries_exp_support.csv"), sep="\t")

    texts = list(df["text"])
    ids = list(df["id"])

    texts_shingls = shinglorizer(texts)
    vectors = vectorizer(texts_shingls)
    searcher.add(ids, texts, vectors)

    """2) Поиск похожих текстов:"""
    searched_texts = ["учетная политика 2022", "что изменить в учетной политике на 2022 г",
                      "как составить пояснения в налоговую инспекцию образцы ответов"]

    searched_shingls = shinglorizer(searched_texts)
    searched_vectors = vectorizer(searched_shingls)
    result = searcher.search(vectors=searched_vectors,  score=0.7)
    print(result)

