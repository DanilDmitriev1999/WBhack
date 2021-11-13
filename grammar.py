from natasha import Segmenter, MorphVocab, NewsEmbedding, NewsMorphTagger, Doc
segmenter = Segmenter()
morphology_vocab = MorphVocab()
embeddings = NewsEmbedding()
morphology_tagger = NewsMorphTagger(embeddings)

def remove_verbs(text: str) -> str:
    natasha_text = Doc(text)
    natasha_text.segment(segmenter)

    natasha_text.tag_morph(morphology_tagger)

    result = []
    for token in natasha_text.tokens:
        if token.pos != 'VERB':
            result.append(token.text)

    return " ".join(result)

def clean_text(text: str):
    """Чистим текст от случайных символов в выдаче"""
    text = text.replace('\\', "")
    
    return text.strip()