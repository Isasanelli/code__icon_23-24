import pandas as pd
import numpy as np
import spacy
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences

class Embedding:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_md")

    def build_embedding(self, original_sentences):
        MAX_WORDS = 25000
        MAX_SEQUENCE_LENGTH = 80

        tokenizer = Tokenizer(num_words=MAX_WORDS, char_level=False)
        tokenizer.fit_on_texts(original_sentences)

        sequences = tokenizer.texts_to_sequences(original_sentences)
        padded_sequences = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post')

        embeddings = []
        for sentence in padded_sequences:
            doc = " ".join([tokenizer.index_word.get(tok, "") for tok in sentence if tok != 0])
            doc_embedding = self.nlp(doc).vector
            embeddings.append(doc_embedding)

        return np.array(embeddings)
