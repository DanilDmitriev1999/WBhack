import streamlit as st
from indexer import FAISS
from annotated_text import annotated_text

indexer_vector_dim = 384
indexer = FAISS(indexer_vector_dim)
indexer.init_index()
indexer.init_vectors()

st.header('Модель предложения тэгов')
x = st.text_input(label='Введите запрос')

if x != '':
    st.write(" ".join(indexer.suggest_tags(x)))