"""NLP features: sentiment analysis, topic modeling, multilingual support."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Optional heavy deps - imported lazily so the app still runs without them
try:
    from textblob import TextBlob
    _HAS_TEXTBLOB = True
except Exception:
    _HAS_TEXTBLOB = False

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _HAS_VADER = True
except Exception:
    _HAS_VADER = False

try:
    from langdetect import detect, DetectorFactory
    DetectorFactory.seed = 0
    _HAS_LANGDETECT = True
except Exception:
    _HAS_LANGDETECT = False

try:
    from deep_translator import GoogleTranslator
    _HAS_TRANSLATOR = True
except Exception:
    _HAS_TRANSLATOR = False


class SentimentAnalyzer:
    """VADER-backed sentiment analyzer (falls back to TextBlob)."""

    def __init__(self):
        self._vader = None
        if _HAS_VADER:
            self._vader = SentimentIntensityAnalyzer()

    def analyze(self, text: str) -> dict:
        if not text or not str(text).strip():
            return {"polarity": 0.0, "subjectivity": 0.0, "label": "neutral"}
        text = str(text)

        if self._vader:
            scores = self._vader.polarity_scores(text)
            compound = scores["compound"]
            if compound >= 0.05:
                label = "positive"
            elif compound <= -0.05:
                label = "negative"
            else:
                label = "neutral"
            return {
                "polarity": float(compound),
                "subjectivity": 0.0,
                "label": label,
                "pos": float(scores["pos"]),
                "neu": float(scores["neu"]),
                "neg": float(scores["neg"]),
            }

        if _HAS_TEXTBLOB:
            blob = TextBlob(text)
            polarity = float(blob.sentiment.polarity)
            subjectivity = float(blob.sentiment.subjectivity)
            if polarity > 0.1:
                label = "positive"
            elif polarity < -0.1:
                label = "negative"
            else:
                label = "neutral"
            return {"polarity": polarity, "subjectivity": subjectivity, "label": label}

        return {"polarity": 0.0, "subjectivity": 0.0, "label": "neutral"}

    def analyze_dataframe(self, df: pd.DataFrame, column: str = "description") -> pd.DataFrame:
        df = df.copy()
        results = df[column].fillna("").apply(self.analyze).apply(pd.Series)
        for col in results.columns:
            df[f"{column}_{col}"] = results[col]
        return df


class TopicModeler:
    """LDA or NMF topic modeling on resource descriptions."""

    def __init__(self, n_topics: int = 5, method: str = "lda"):
        self.n_topics = n_topics
        self.method = method.lower()
        self.model = None
        self.vectorizer = None
        self.feature_names = None
        self.topic_keywords_ = None

    def fit(self, texts):
        if self.method == "lda":
            self.vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words="english")
            doc_term = self.vectorizer.fit_transform(texts)
            self.model = LatentDirichletAllocation(
                n_components=self.n_topics, random_state=42, learning_method="batch"
            )
        else:  # nmf
            self.vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words="english")
            doc_term = self.vectorizer.fit_transform(texts)
            self.model = NMF(n_components=self.n_topics, random_state=42)

        self.model.fit(doc_term)
        self.feature_names = self.vectorizer.get_feature_names_out()
        self._extract_topic_keywords()
        return self

    def _extract_topic_keywords(self, top_words: int = 8):
        self.topic_keywords_ = []
        for topic_idx, topic in enumerate(self.model.components_):
            top_indices = topic.argsort()[:-top_words - 1:-1]
            keywords = [self.feature_names[i] for i in top_indices]
            self.topic_keywords_.append(keywords)

    def transform(self, texts):
        if self.vectorizer is None:
            raise ValueError("Call fit() first")
        doc_term = self.vectorizer.transform(texts)
        return self.model.transform(doc_term)

    def assign_topics(self, df: pd.DataFrame, column: str = "description") -> pd.DataFrame:
        texts = df[column].fillna("").tolist()
        df = df.copy()
        if not any(t.strip() for t in texts):
            df["discovered_topic"] = "unknown"
            df["topic_confidence"] = 0.0
            return df
        # Fit the model if it hasn't been fitted yet
        if self.model is None:
            self.fit(texts)
        doc_topics = self.transform(texts)
        top_topic = doc_topics.argmax(axis=1)
        confidence = doc_topics.max(axis=1)
        topic_names = []
        for i in top_topic:
            if self.topic_keywords_:
                kw = ", ".join(self.topic_keywords_[i][:3])
                topic_names.append(f"Topic {i}: {kw}")
            else:
                topic_names.append(f"Topic {i}")
        df["discovered_topic"] = topic_names
        df["topic_confidence"] = confidence
        return df


class MultilingualSupport:
    """Language detection and translation helpers."""

    @staticmethod
    def detect_language(text: str) -> str:
        if not text or not str(text).strip():
            return "unknown"
        if not _HAS_LANGDETECT:
            return "en"
        try:
            return detect(str(text))
        except Exception:
            return "unknown"

    @staticmethod
    def translate_to_english(text: str, source_lang: Optional[str] = None) -> str:
        if not text or not str(text).strip():
            return ""
        if not _HAS_TRANSLATOR:
            return str(text)
        try:
            if source_lang is None:
                source_lang = MultilingualSupport.detect_language(text)
            if source_lang == "en" or source_lang == "unknown":
                return str(text)
            return GoogleTranslator(source=source_lang, target="en").translate(str(text))
        except Exception:
            return str(text)

    @staticmethod
    def detect_dataframe(df: pd.DataFrame, column: str = "description") -> pd.DataFrame:
        df = df.copy()
        df[f"{column}_lang"] = df[column].fillna("").apply(MultilingualSupport.detect_language)
        return df