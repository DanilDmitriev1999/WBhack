import gzip
import pandas as pd

def read_query_data():
    with gzip.open('query_popularity.csv.gz') as f:
        query_popularity = pd.read_csv(f, sep='\t', escapechar='\\')

    return query_popularity
