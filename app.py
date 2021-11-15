import streamlit as st
from indexer import FAISS
# from annotated_text import annotated_text
import pymorphy2

morph = pymorphy2.MorphAnalyzer()
indexer_vector_dim = 384
indexer = FAISS(indexer_vector_dim)
indexer.init_index('new_index.index')
indexer.init_vectors('new_vectors.pkl')


def main():
    st.set_page_config(layout="wide")
    st.header('Модель предложения тэгов')

    x = st.text_input(label='Введите запрос')
    if x != '':
        # result = indexer.suggest_tags(x)
        st.write(" / ".join(indexer.suggest_tags(x)))
        # cols = st.beta_columns((1, 1, 1, 1, 1, 1, 1, 1, 1, 1))
        # for i in range(len(result)):
            #     cols[i].subheader(result[i])


if __name__ == '__main__':
    main()