import pickle
import faiss
import numpy as np
import pandas as pd
from utils import *
from sentence_transformers import SentenceTransformer

from tqdm import tqdm
from typing import List


class FAISS:
    def __init__(self, dimensions: int) -> None:
        self.dimensions = dimensions
        self.index = faiss.IndexFlatL2(dimensions)
        self.vectors = {}
        self.counter = 0
        self.model_name = 'paraphrase-multilingual-MiniLM-L12-v2'
        self.sentence_encoder = SentenceTransformer(self.model_name)

    def init_vectors(self, path: str) -> None:
        """
        Заполняет набор векторов предобученными значениями

        Args:
            path: путь к файлу в формате pickle
        """
        with open(path, 'rb') as pkl_file:
            self.vectors = pickle.load(pkl_file)

            self.counter = len(self.vectors)

    def init_index(self, path) -> None:
        """
        Заполняет индекс FAISS предобученными значениями

        Args:
            path: путь к файлу в формате FAISS
        """
        self.index = faiss.read_index(path)
    
    def save_vectors(self, path: str) -> None:
        """
        Сохраняет набор векторов

        Args:
            path: желаемый путь к файлу
        """
        with open(path, "wb") as fp:
            pickle.dump(self.index.vectors, fp)

    def save_index(self, path: str) -> None:
        """
        Сохраняет индекс FAISS

        Args:
            path: желаемый путь к файлу
        """
        faiss.write_index(self.index, path)

    def add(self, text: str, idx: int, pop: float, emb=None) -> None:
        """
        Добавляет в поисковый индекс новый вектор

        Args:
            text: текст запроса
            idx: индекс нового вектора
            pop: популярность запроса
            emb (optional): эмбеддинг текста запроса (если не указан, то будет подготовлен с помощью self.sentence_encoder)
        """
        if emb is None:
            text_vec = self.sentence_encoder.encode([text])
        else:
            text_vec = emb
    
        self.index.add(text_vec)
        self.vectors[self.counter] = (idx, text, pop, text_vec)

        self.counter += 1

    def search(self, v: List, k: int = 10) -> List[List]:
        """
        Ищет в поисковом индексе ближайших соседей к вектору v

        Args:
            v: вектор для поиска ближайших соседей
            k: число векторов в выдаче
        Returns:
            список векторов, ближайших к вектору v, в формате [idx, text, popularity, similarity]
        """
        result = []
        distance, item_index = self.index.search(v, k)
        for dist, i in zip(distance[0], item_index[0]):
            if i == -1:
                break
            else:
                result.append((self.vectors[i][0], self.vectors[i][1], self.vectors[i][2], dist))

        return result

    def suggest_tags(self, query: str, top_n: int = 10, k: int = 30) -> List[str]:
        """
        Получает список тегов для пользователя по текстовому запросу

        Args:
            query: запрос пользователя
            top_n (optional): число тегов в выдаче
            k (optional): число векторов из индекса, среди которых будут искаться теги для выдачи
        Returns:
            список тегов для выдачи пользователю
        """
        emb = self.sentence_encoder.encode([query.lower()])
        r = self.search(emb, k)

        result = []
        for i in r:
            if check(query, i[1]):
                result.append(i)
        # надо добавить вес относительно длины
        result = sorted(result, key=lambda x: x[0] * 0.3 - x[-1], reverse=True)
        total_result = []
        for i in range(len(result)):
            flag = True
            for j in result[i + 1:]:
                flag &= easy_check(result[i][1], j[1])
            if flag:
                total_result.append(result[i][1])

        return total_result[:top_n]

    def fill(self, queries: List[str], popularities: pd.DataFrame) -> None:
        """
        Заполняет поисковый индекс запросами queries, популярности которых берутся из таблицы popularities

        Args:
            queries: список запросов
            popularities: таблица, в которой содержатся колонки query и query_popularity
        """
        idx = -1
        for query in tqdm(queries):
            idx += 1
            if type(query) == str:
                emb = self.index.sentence_encoder.encode([query.lower()])
                bool_add = True
                search_sim = self.index.search(emb, 1)

                try:
                    popularity = popularities[popularities["query"] == query]["query_popularity"].item()
                except ValueError:
                    # Если для текущего запроса неизвестна популярность, возьмем значение 5
                    popularity = 5

                if len(search_sim) > 0:
                    search_sim = search_sim[0]
                    if search_sim[-1] < 0.15:
                        # Не добавляем вектор, если он находится достаточно близко к уже присутствующему в индексе
                        bool_add = False
                    if bool_add:
                        self.index.add(query, popularity, idx, emb)  
                else:
                    self.index.add(query, popularity, idx, emb)