import json
import re
from pathlib import Path
from typing import Dict, Tuple

import nbformat as nbf

COURSE_INDEX = Path("course_index.json")
NOTEBOOK_ROOT = Path("notebooks")


LOGREG_DEMO = """\
# Logistic regression on synthetic binary data
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

X, y = make_classification(n_samples=400, n_features=6, random_state=0)
clf = LogisticRegression(max_iter=500, solver="lbfgs").fit(X, y)
pred = clf.predict(X)
print(classification_report(y, pred, digits=3))
"""

MLP_DEMO = """\
# Two-layer neural network on a toy dataset
from sklearn.datasets import make_moons
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

X, y = make_moons(noise=0.2, random_state=0)
mlp = MLPClassifier(hidden_layer_sizes=(16, 16), max_iter=500, random_state=0)
mlp.fit(X, y)
print("Accuracy:", accuracy_score(y, mlp.predict(X)))
"""
ATTENTION_DEMO = """\
# Toy scaled dot-product attention
import numpy as np
np.random.seed(0)

def attention(query, key, value):
    scores = query @ key.T / np.sqrt(key.shape[-1])
    weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
    weights = weights / weights.sum(axis=-1, keepdims=True)
    return weights @ value

Q = np.random.randn(2, 4)
K = np.random.randn(5, 4)
V = np.random.randn(5, 3)
context = attention(Q, K, V)
print("Context shape:", context.shape)
print(context)
"""

NGRAM_DEMO = """\
# Quick bigram language model demo
from collections import Counter
import math

corpus = "we love NLP and we love transformers".lower().split()
bigrams = Counter(zip(corpus, corpus[1:]))
unigrams = Counter(corpus)


def bigram_prob(w1, w2, alpha=1.0):
    return (bigrams[(w1, w2)] + alpha) / (unigrams[w1] + alpha * len(unigrams))


sentence = "we love nlp".split()
log_prob = 0.0
for w1, w2 in zip(sentence, sentence[1:]):
    log_prob += math.log(bigram_prob(w1, w2))
print(f"log P({' '.join(sentence)}) = {log_prob:.3f}")
"""

BEAM_DEMO = """\
# Minimal beam search over a toy vocabulary
vocab = {
    "<s>": {"i": -0.1, "we": -0.3},
    "i": {"love": -0.2, "like": -0.4},
    "we": {"love": -0.3, "enjoy": -0.5},
    "love": {"nlp": -0.1, "</s>": -1.0},
    "like": {"nlp": -0.2, "</s>": -1.2},
    "enjoy": {"transformers": -0.2, "</s>": -1.1},
}

beam = [("<s>", 0.0)]
beam_size = 2

for _ in range(3):
    candidates = []
    for seq, score in beam:
        last = seq.split()[-1]
        for word, logp in vocab.get(last, {}).items():
            candidates.append((seq + " " + word, score + logp))
    beam = sorted(candidates, key=lambda x: x[1])[:beam_size]

for seq, score in beam:
    print(f"{seq}  (logP={score:.2f})")
"""

NUCLEUS_DEMO = """\
# Nucleus sampling illustration
import numpy as np

logits = np.array([2.3, 1.2, 0.7, 0.1, -0.5])
probs = np.exp(logits - logits.max())
probs /= probs.sum()

sorted_idx = np.argsort(probs)[::-1]
sorted_probs = probs[sorted_idx]
cumulative = np.cumsum(sorted_probs)
p = 0.9
cutoff = np.searchsorted(cumulative, p) + 1

support = sorted_idx[:cutoff]
top_probs = sorted_probs[:cutoff]
top_probs /= top_probs.sum()

sample = np.random.choice(support, p=top_probs)
print("Support after nucleus filter:", support)
print("Sampled token id:", int(sample))
"""

EMBEDDING_DEMO = """\
# Toy document embeddings with SVD
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
import numpy as np

docs = [
    "language models generate text",
    "neural networks learn representations",
    "word embeddings capture similarity",
    "transformers attend to context",
]
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(docs)
svd = TruncatedSVD(n_components=2, random_state=0).fit_transform(X)
for doc, vec in zip(docs, svd):
    print(doc, "->", np.round(vec, 3))
"""

HMM_DEMO = """\
# Tiny HMM forward probabilities
states = ["Rainy", "Sunny"]
start = {"Rainy": 0.6, "Sunny": 0.4}
trans = {
    "Rainy": {"Rainy": 0.7, "Sunny": 0.3},
    "Sunny": {"Rainy": 0.4, "Sunny": 0.6},
}
emit = {
    "Rainy": {"walk": 0.1, "shop": 0.4, "clean": 0.5},
    "Sunny": {"walk": 0.6, "shop": 0.3, "clean": 0.1},
}
obs = ["walk", "shop", "clean"]

alpha = [{s: start[s] * emit[s][obs[0]] for s in states}]
for t in range(1, len(obs)):
    alpha.append({})
    for s in states:
        alpha[t][s] = emit[s][obs[t]] * sum(
            alpha[t - 1][sp] * trans[sp][s] for sp in states
        )

for t, dist in enumerate(alpha):
    print(f"t={t}", {s: round(p, 4) for s, p in dist.items()})
"""

DEFAULT_DEMO = """\
print('Run your own mini-experiment here—consider adding code that reinforces the lecture concepts!')
"""


def slugify(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-") or "item"


def choose_demo(title: str) -> Tuple[str, str]:
    lower = title.lower()
    if "neural" in lower or "mlp" in lower:
        return MLP_DEMO, "Train a tiny two-layer neural network on make_moons."
    if any(key in lower for key in ["logistic", "perceptron", "linear", "sentiment", "classification"]):
        return LOGREG_DEMO, "Logistic regression on a synthetic classification task."
    if any(key in lower for key in ["attention", "self-attention", "transformer", "multi-head"]):
        return ATTENTION_DEMO, "Scaled dot-product attention with random toy inputs."
    if any(key in lower for key in ["n-gram", "language model", "perplexity", "smoothing"]):
        return NGRAM_DEMO, "Estimate simple bigram probabilities from a toy corpus."
    if any(key in lower for key in ["beam", "decoding"]):
        return BEAM_DEMO, "Beam search over a tiny vocabulary."
    if any(key in lower for key in ["nucleus", "sampling", "temperature"]):
        return NUCLEUS_DEMO, "Nucleus (top-p) sampling on a toy logit distribution."
    if any(key in lower for key in ["embedding", "skip-gram", "word2vec"]):
        return EMBEDDING_DEMO, "Project simple bag-of-words vectors into a low-dimensional space."
    if any(key in lower for key in ["hmm", "markov", "pos", "sequence labeling", "viterbi"]):
        return HMM_DEMO, "Forward pass probabilities for a toy hidden Markov model."
    return DEFAULT_DEMO, "Placeholder—swap in a quick experiment that matches the lecture."


def locate_notebook(week: Dict) -> Dict[str, Path]:
    title = week.get("week_title", "")
    match = re.match(r"Week\s+(\d+):\s*(.*)", title)
    if match:
        folder = f"{int(match.group(1)):02d}-{slugify(match.group(2))}"
    else:
        folder = slugify(title)
    base = NOTEBOOK_ROOT / folder
    mapping = {}
    for item in week.get("items", []):
        if item.get("type") != "video":
            continue
        fname = base / f"{slugify(item.get('title', 'item'))}.ipynb"
        mapping[item.get("title", "")] = fname
    return mapping


def update_notebook(path: Path, title: str) -> bool:
    if not path.exists():
        return False
    demo_code, demo_caption = choose_demo(title)
    nb = nbf.read(path, as_version=4)
    changed = False

    indices_to_remove = [
        idx
        for idx, cell in enumerate(nb.cells)
        if cell.cell_type == "code"
        and "random.seed" in cell.source
        and "CI" in cell.source
    ]
    for idx in reversed(indices_to_remove):
        nb.cells.pop(idx)
        changed = True

    demo_updated = False
    for idx, cell in enumerate(nb.cells):
        if cell.cell_type == "markdown" and cell.source.strip().startswith("## Demo"):
            new_source = f"## Demo\n{demo_caption}"
            if cell.source != new_source:
                cell.source = new_source
                changed = True
            for j in range(idx + 1, len(nb.cells)):
                next_cell = nb.cells[j]
                if next_cell.cell_type == "code":
                    if next_cell.source != demo_code:
                        next_cell.source = demo_code
                        changed = True
                    demo_updated = True
                    break
            break

    if not demo_updated:
        nb.cells.append(nbf.v4.new_markdown_cell(f"## Demo\n{demo_caption}"))
        nb.cells.append(nbf.v4.new_code_cell(demo_code))
        changed = True

    if changed:
        nbf.write(nb, path)
    return changed


def main() -> None:
    idx = json.loads(COURSE_INDEX.read_text(encoding="utf-8"))
    updated = 0
    missing = []
    for week in idx.get("weeks", []):
        notebook_map = locate_notebook(week)
        for title, path in notebook_map.items():
            if update_notebook(path, title):
                updated += 1
            elif not path.exists():
                missing.append(title)
    print(f"Updated {updated} notebooks with demo snippets.")
    if missing:
        print("Missing notebooks for:")
        for title in missing:
            print(" -", title)


if __name__ == "__main__":
    main()
