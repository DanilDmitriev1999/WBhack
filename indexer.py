import pickle
import faiss
import numpy as np
from grammar import remove_verbs, clean_text
from sentence_transformers import SentenceTransformer

class FAISS:
    def __init__(self, dimensions: int):
        self.dimensions = dimensions
        self.index = faiss.IndexFlatL2(dimensions)
        self.vectors = {}
        self.counter = 0
        self.model_name = 'paraphrase-multilingual-MiniLM-L12-v2'
        self.sentence_encoder = SentenceTransformer(self.model_name)
    
    def init_vectors(self):
        with open('data/data.pkl', 'rb') as pkl_file:
            self.vectors = pickle.load(pkl_file)
        
    def init_index(self):
        self.index = faiss.read_index('data/vector.index')
    
    def add(self, text: str, idx: int):
        text_vec = self.sentence_encoder.encode([text])
        try:
            self.index.add(text_vec)
            self.vectors[self.counter] = (idx, text, text_vec)
            self.counter += 1
        except:
            print(f'text: {text}')
            print(f'text_vec: {text_vec}')
            
        
    def search(self, v: list, k: int=10) -> list:
        result = []
        distance, item_index = self.index.search(v, k)
        for dist, i in zip(distance[0], item_index[0]):
            if i == -1:
                break
            else:
                result.append((self.vectors[i][0], self.vectors[i][1], dist))
                    
        return result

    def suggest_tags(self, query, k=10) -> list: 
        embedding = self.sentence_encoder.encode([query])
        tags = self.search(embedding, k)

        suggestion = []

        for tag in tags:
            suggestion.append(clean_text(remove_verbs(tag[1])))

        return suggestion