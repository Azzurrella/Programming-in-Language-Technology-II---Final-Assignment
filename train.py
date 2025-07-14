import os
import glob
import pickle
import xml.etree.ElementTree as ET

import spacy
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

# ─── Helpers for pickling ───────────────────────────────────────────────────────

def extract_text(X):
    """Given X = [(text, meta_dict), ...], return [text, ...]."""
    return [t for t, _ in X]

def extract_meta(X):
    """Given X = [(text, meta_dict), ...], return [meta_dict, ...]."""
    return [m for _, m in X]

# ─── Your existing classes ──────────────────────────────────────────────────────

class XMLLoader:
    def __init__(self, paths):
        self.paths = paths

    def load(self):
        X_text, X_meta, y = [], [], []
        for path in self.paths:
            tree = ET.parse(path)
            root = tree.getroot()
            for sentence in root.findall('.//sentence'):
                text = sentence.find('text').text.strip()
                opinions = sentence.find('Opinion') or sentence.find('Opinions')
                if opinions is None:
                    continue
                for op in sentence.findall('.//Opinion'):
                    X_text.append(text)
                    X_meta.append({'category': op.get('category')})
                    y.append(op.get('polarity'))
        return X_text, X_meta, y

class POSFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, model='en_core_web_sm'):
        try:
            self.nlp = spacy.load(model, disable=["ner", "parser"])
        except OSError:
            from spacy.cli import download
            download(model)
            self.nlp = spacy.load(model, disable=["ner", "parser"])
    def fit(self, X, y=None): return self
    def transform(self, X):
        return [' '.join([t.pos_ for t in doc])
                for doc in self.nlp.pipe(X, batch_size=50)]

class NERCount(BaseEstimator, TransformerMixin):
    def __init__(self, model='en_core_web_sm'):
        try:
            self.nlp = spacy.load(model, disable=["parser"])
        except OSError:
            from spacy.cli import download
            download(model)
            self.nlp = spacy.load(model, disable=["parser"])
    def fit(self, X, y=None): return self
    def transform(self, X):
        return [{'ner_count': len(doc.ents)}
                for doc in self.nlp.pipe(X, batch_size=50)]

# ─── The train_model function ───────────────────────────────────────────────────

def train_model(parts_dir, model_path='absa_model.pkl'):
    # 1) find all your XML files
    xml_paths = glob.glob(os.path.join(parts_dir, 'part*.xml'))
    if not xml_paths:
        raise FileNotFoundError(f"No XML files found in {parts_dir!r}")

    # 2) load data, *before* zipping
    X_text, X_meta, y = XMLLoader(xml_paths).load()

    # 3) combine into a single X
    X = list(zip(X_text, X_meta))

    # 4) sub‐pipelines
    text_pipe = Pipeline([
        ('get_text', FunctionTransformer(extract_text, validate=False)),
        ('tfidf',    TfidfVectorizer(ngram_range=(1,3), min_df=2)),
    ])
    pos_pipe = Pipeline([
        ('get_text', FunctionTransformer(extract_text, validate=False)),
        ('pos',      POSFeatures()),
        ('vect',     CountVectorizer(ngram_range=(1,2))),
    ])
    cat_pipe = Pipeline([
        ('get_meta', FunctionTransformer(extract_meta, validate=False)),
        ('vect',     DictVectorizer()),
    ])
    ner_pipe = Pipeline([
        ('get_text', FunctionTransformer(extract_text, validate=False)),
        ('ner',      NERCount()),
        ('vect',     DictVectorizer()),
    ])

    features = FeatureUnion([
        ('tfidf', text_pipe),
        ('pos',   pos_pipe),
        ('cat',   cat_pipe),
        ('ner',   ner_pipe),
    ])

    # 5) classifier
    pipeline = Pipeline([
        ('features', features),
        ('clf',      LinearSVC(C=1.0, max_iter=20000)),
    ])

    print(f"Training on {len(y)} instances…")
    pipeline.fit(X, y)

    # 6) save
    with open(model_path, 'wb') as f:
        pickle.dump(pipeline, f)
    print(f"Model saved to {model_path!r}")

# ─── Entrypoint ──────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    # first arg: data directory; second arg: model output path
    train_model('./output_parts/', './model')
