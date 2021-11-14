# Система предсказаний поисковых тегов

## Структура проекта

Для того, чтобы вам было проще ориентироваться в репозитории, ниже приведён список файлов и их назначение:

- `app.py` - основной файл приложения для HF Spaces
- `indexer.py` - реализация класса для работы с FAISS (заполнение индекса, поиск, подбор тегов)
- `new_index.index`, `new_vectors.pkl` - файлы для инициализации модели обученными значениями
- `utils.py` - вспомогательные функции (отбор тегов)

## Примеры использования

### Инициализация индекса готовыми значениями

```python
from indexer import FAISS

indexer_vector_dim = 384
indexer = FAISS(indexer_vector_dim)
indexer.init_index('new_index.index')
indexer.init_vectors('new_vectors.pkl')
```

### Запрос к индексу

```python
indexer.suggest_tags("куртка оверсайз")
```

### Заполнение индекса

Пусть у нас есть список запросов `queries` и `DataFrame` `popularities`, в котором присутствуют поля `query` и `query_popularity`, тогда мы можем самостоятельно заполнить поисковый индекс:

```python
indexer.fill(queries, popularities)
```

### Сохранение заполненного индекса

Если мы не хотим потерять заполненные данные, их можно сохранить и затем переиспользовать:

```python
indexer.save_vectors("my_vectors.pkl")
indexer.save_index("my_index.index")
```

# Запуск

```bash
streamlit run app.py
```