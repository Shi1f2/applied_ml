# Stage B: binary sentiment classifiers (pos/neg) trained only on rows that
# Stage A flagged as non-spam. Three feature families are compared:
#   - TF-IDF unigrams+bigrams (sparse, lexical)
#   - Word2Vec trained from scratch on the provided corpus
#   - Pretrained GloVe embeddings (transfer features)
import numpy as np
import gensim.downloader as gensim_downloader
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from .preprocess import tokenize, tokenize_corpus


class TfidfSentiment:
    def __init__(self, random_state=42):
        self.vectorizer = TfidfVectorizer(
            tokenizer=tokenize,
            token_pattern=None,
            ngram_range=(1, 2),
            min_df=3,
            max_df=0.95,
            sublinear_tf=True,
        )
        self.clf = LogisticRegression(max_iter=2000, C=1.0, random_state=random_state)

    def fit(self, texts, labels):
        matrix = self.vectorizer.fit_transform(texts)
        self.clf.fit(matrix, labels)
        return self

    def predict(self, texts):
        return self.clf.predict(self.vectorizer.transform(texts))

    def predict_proba(self, texts):
        return self.clf.predict_proba(self.vectorizer.transform(texts))[:, 1]


class EmbeddingSentiment:
    def __init__(self, vector_size=200, window=5, min_count=3, epochs=10,
                 workers=2, random_state=42):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.epochs = epochs
        self.workers = workers
        self.random_state = random_state
        self.w2v = None
        self.clf = LogisticRegression(max_iter=2000, C=1.0, random_state=random_state)

    def _embed(self, tokenised):
        # Average pooling over in-vocab tokens; OOV-only rows stay as zeros.
        vectors = np.zeros((len(tokenised), self.vector_size), dtype=np.float32)
        wv = self.w2v.wv
        for i, tokens in enumerate(tokenised):
            present = [t for t in tokens if t in wv.key_to_index]
            if present:
                vectors[i] = np.mean(wv[present], axis=0)
        return vectors

    def fit(self, texts, labels):
        tokenised = tokenize_corpus(texts)
        self.w2v = Word2Vec(
            sentences=tokenised,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
            epochs=self.epochs,
            seed=self.random_state,
        )
        features = self._embed(tokenised)
        self.clf.fit(features, labels)
        return self

    def predict(self, texts):
        return self.clf.predict(self._embed(tokenize_corpus(texts)))

    def predict_proba(self, texts):
        return self.clf.predict_proba(self._embed(tokenize_corpus(texts)))[:, 1]


class PretrainedEmbeddingSentiment:
    def __init__(self, model_name='glove-wiki-gigaword-100', random_state=42):
        self.model_name = model_name
        self.random_state = random_state
        self.wv = None
        self.vector_size = None
        self.clf = LogisticRegression(max_iter=2000, C=1.0, random_state=random_state)

    def _embed(self, tokenised):
        vectors = np.zeros((len(tokenised), self.vector_size), dtype=np.float32)
        for i, tokens in enumerate(tokenised):
            present = [t for t in tokens if t in self.wv.key_to_index]
            if present:
                vectors[i] = np.mean(self.wv[present], axis=0)
        return vectors

    def fit(self, texts, labels):
        self.wv = gensim_downloader.load(self.model_name)
        self.vector_size = self.wv.vector_size
        features = self._embed(tokenize_corpus(texts))
        self.clf.fit(features, labels)
        return self

    def predict(self, texts):
        return self.clf.predict(self._embed(tokenize_corpus(texts)))

    def predict_proba(self, texts):
        return self.clf.predict_proba(self._embed(tokenize_corpus(texts)))[:, 1]
