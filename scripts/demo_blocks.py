"""Small, fast demo code blocks keyed by topic keywords."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List


@dataclass(frozen=True)
class DemoBlock:
    title: str
    code: str
    tags: List[str]


_BLOCKS: Dict[str, DemoBlock] = {
    "logistic": DemoBlock(
        title="Logistic regression on toy sentiment",
        tags=["demo", "light"],
        code="""
import os
import random
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

CI = os.environ.get("CI", "").lower() == "true"
texts = [
    "I loved this movie, it was fantastic!",
    "The plot was terrible and boring.",
    "Amazing soundtrack and great acting.",
    "I would not recommend this film to anyone.",
    "A delightful and heartwarming story.",
    "Awful pacing with wooden characters.",
]
labels = [1, 0, 1, 0, 1, 0]

vectorizer = CountVectorizer(stop_words="english")
X = vectorizer.fit_transform(texts)
clf = LogisticRegression(max_iter=100)
clf.fit(X, labels)

feature_names = np.array(vectorizer.get_feature_names_out())
coefs = clf.coef_[0]
order = np.argsort(coefs)
print("Top positive tokens:", feature_names[order][-5:][::-1])
print("Top negative tokens:", feature_names[order][:5])
print("Training accuracy:", clf.score(X, labels))
""",
    ),
    "perceptron": DemoBlock(
        title="Perceptron update dynamics",
        tags=["demo", "light"],
        code="""
import numpy as np
from sklearn.linear_model import Perceptron

X = np.array(
    [
        [1.0, 1.0],
        [1.0, -1.0],
        [-1.0, 1.0],
        [-1.0, -1.0],
    ]
)
y = np.array([1, -1, -1, 1])

model = Perceptron(max_iter=10, eta0=1.0, fit_intercept=True, random_state=0)
model.fit(X, y)
print("Weights:", model.coef_)
print("Intercept:", model.intercept_)
print("Predictions:", model.predict(X))
""",
    ),
    "softmax": DemoBlock(
        title="Softmax regression on synthetic data",
        tags=["demo", "light"],
        code="""
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression

X, y = make_blobs(n_samples=150, centers=3, n_features=2, random_state=2)
clf = LogisticRegression(max_iter=200, multi_class="multinomial")
clf.fit(X, y)
print("Coefficients shape:", clf.coef_.shape)
print("Accuracy:", clf.score(X, y))
""",
    ),
    "attention": DemoBlock(
        title="Scaled dot-product attention",
        tags=["demo", "light"],
        code="""
import torch

queries = torch.randn(2, 4)
keys = torch.randn(3, 4)
values = torch.randn(3, 5)
scale = queries.size(-1) ** 0.5
weights = torch.softmax((queries @ keys.T) / scale, dim=-1)
context = weights @ values
print("Attention weights:\n", weights)
print("Context shape:", context.shape)
""",
    ),
    "positional": DemoBlock(
        title="Visualising sinusoidal positional encodings",
        tags=["demo", "light"],
        code="""
import math
import matplotlib.pyplot as plt
import numpy as np

positions = np.arange(0, 100)
d_model = 32
angles = np.zeros((len(positions), d_model))
for pos in positions:
    for i in range(0, d_model, 2):
        angle = pos / (10000 ** (i / d_model))
        angles[pos, i] = math.sin(angle)
        angles[pos, i + 1] = math.cos(angle)

plt.figure(figsize=(8, 4))
plt.plot(positions, angles[:, :4])
plt.title("Sinusoidal positional encodings")
plt.xlabel("Position")
plt.ylabel("Encoding value")
plt.show()
""",
    ),
    "bert": DemoBlock(
        title="Sentiment analysis with a pretrained BERT pipeline",
        tags=["demo", "inference"],
        code="""
import os
from transformers import pipeline

CI = os.environ.get("CI", "").lower() == "true"
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
classifier = pipeline("sentiment-analysis", model=model_name)
result = classifier("The lecture made transformer attention crystal clear!")
print(result)
""",
    ),
    "bart": DemoBlock(
        title="Summarisation with BART",
        tags=["demo", "inference", "heavy"],
        code="""
import os
from transformers import pipeline

if os.environ.get("CI", "").lower() == "true":
    raise SystemExit("Skipping heavy summarisation demo in CI")

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
text = (
    "Transformers use self-attention mechanisms to weigh the influence of different tokens "
    "when encoding sequences. Positional encodings inject order information into otherwise "
    "permutation-invariant architectures."
)
print(summarizer(text, max_length=45, min_length=10))
""",
    ),
    "bleu": DemoBlock(
        title="Machine translation with sacrebleu scoring",
        tags=["demo", "inference"],
        code="""
from sacrebleu import corpus_bleu
from transformers import pipeline

translator = pipeline("translation_en_to_de", model="Helsinki-NLP/opus-mt-en-de")
reference = ["Das ist ein Test."]
translation = translator("This is a test.")[0]["translation_text"]
print("Translation:", translation)
print("BLEU:", corpus_bleu([translation], [reference]).score)
""",
    ),
    "qa": DemoBlock(
        title="Question answering pipeline",
        tags=["demo", "inference"],
        code="""
from transformers import pipeline

qa_pipe = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
context = "The NLP course covers transformers, attention, and modern language models."
question = "What topics does the course cover?"
print(qa_pipe(question=question, context=context))
""",
    ),
    "lime": DemoBlock(
        title="Explaining predictions with LIME",
        tags=["demo", "analysis"],
        code="""
import numpy as np
from lime.lime_text import LimeTextExplainer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

texts = [
    "This class is excellent and insightful.",
    "The explanation lacked depth and was confusing.",
    "I appreciate the clear intuition in the lecture.",
    "The material feels rushed and hard to follow.",
]
labels = np.array([1, 0, 1, 0])
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)
model = LogisticRegression(max_iter=200)
model.fit(X, labels)

explainer = LimeTextExplainer(class_names=["negative", "positive"])
exp = explainer.explain_instance(
    "The lecture was engaging and accessible.",
    lambda xs: np.stack([model.predict_proba(vectorizer.transform([x]))[0] for x in xs]),
    num_features=5,
)
print(exp.as_list())
""",
    ),
}


_KEYWORDS = {
    "perceptron": ["perceptron"],
    "logistic": ["logistic", "classification", "intro", "course preview"],
    "softmax": ["softmax", "multiclass"],
    "attention": ["attention", "self-attention"],
    "positional": ["positional", "position"],
    "bert": ["bert", "fine-tuning", "pre-training"],
    "bart": ["bart", "summarization"],
    "bleu": ["mt", "translation"],
    "qa": ["qa", "dialogue", "question"],
    "lime": ["explanation", "interpretability", "explanations"],
}


def pick_blocks_for(title: str) -> Iterable[DemoBlock]:
    title_lower = title.lower()
    for key, keywords in _KEYWORDS.items():
        if any(word in title_lower for word in keywords):
            yield _BLOCKS[key]

