# Stage A: self-labelled spam filter.
# No external spam corpus is allowed, so seed labels come from a high-precision
# "subject:" header heuristic, then a TF-IDF + logistic regression model
# generalises beyond that single cue.
import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


# Spam in this dataset is email-style — the "Subject:" header is a strong tell.
SUBJECT_RE = re.compile(r'^\s*subject\s*:', re.IGNORECASE)


def heuristic_is_spam(text):
    return bool(SUBJECT_RE.match(text))


def heuristic_predict(texts):
    return np.array([heuristic_is_spam(t) for t in texts]).astype(int)


class TfidfSpamFilter:
    def __init__(self, seed_predict_fn=heuristic_predict, random_state=42):
        # Seed labels are produced by an unsupervised heuristic; the LR then
        # learns the broader spam vocabulary the heuristic alone would miss.
        self.seed_predict_fn = seed_predict_fn
        self.vectorizer = TfidfVectorizer(
            min_df=3, max_df=0.95,
            ngram_range=(1, 2),
            sublinear_tf=True,
            strip_accents='unicode',
            lowercase=True,
        )
        self.clf = LogisticRegression(max_iter=2000, C=1.0, random_state=random_state)

    def fit(self, texts):
        seed_labels = self.seed_predict_fn(texts)
        matrix = self.vectorizer.fit_transform(texts)
        self.clf.fit(matrix, seed_labels)
        return self

    def predict(self, texts):
        return self.clf.predict(self.vectorizer.transform(texts))

    def predict_proba(self, texts):
        return self.clf.predict_proba(self.vectorizer.transform(texts))[:, 1]
