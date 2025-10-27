import json
import re
from pathlib import Path
from typing import Dict, Tuple

import nbformat as nbf

NOTES_PATH = Path("ut austin nlp.txt")
COURSE_INDEX = Path("course_index.json")
NOTEBOOK_ROOT = Path("notebooks")

DOMAIN_PATTERN = re.compile(
    r"((?:cs|www|aclanthology|arxiv|dl|colah|reuters|github|ar5iv)[^\s]*)",
    re.I,
)

FOOTNOTE_PATTERN = re.compile(r"【[^】]+】")
WHITESPACE_PATTERN = re.compile(r"\s+")

FALLBACK_SUMMARIES: Dict[str, Dict[str, str]] = {
    "bert-for-qa": {
        "title": "BERT for QA",
        "summary": (
            "Demonstrates how bidirectional transformer encoders are fine-tuned for span extraction tasks like SQuAD. "
            "BERT processes the question and passage jointly, produces contextual token embeddings, then trains two classifiers to predict start and end positions of the answer. "
            "The lecture walks through input packing with [CLS] and [SEP], the objective function, and why contextualized attention is especially good at finding answer spans compared to earlier RNNs. "
            "It also highlights best practices such as layer-wise learning rates, handling unanswerable questions, and evaluation with exact match and F1."
        ),
    },
    "cross-lingual-pre-training": {
        "title": "Cross-lingual Pre-training",
        "summary": (
            "Covers multilingual language models such as mBERT, XLM, and XLM-R that are trained on large corpora spanning dozens of languages. "
            "Students see how shared vocabulary, joint subword tokenization, and masked language modeling objectives let the model learn aligned semantic spaces without parallel data. "
            "The video surveys zero-shot transfer results on XNLI, POS tagging, and QA, and explains practical tips like vocabulary balancing and language adaptive fine-tuning. "
            "It closes with open challenges around low-resource languages, script diversity, and catastrophic forgetting when adapting to a new language."
        ),
    },
    "cross-lingual-tagging-and-parsing": {
        "title": "Cross-lingual Tagging and Parsing",
        "summary": (
            "Explains strategies for building sequence taggers and dependency parsers when labeled data exists only in a source language. "
            "Methods include annotation projection through parallel corpora, model transfer with multilingual embeddings, and multi-source ensembling. "
            "Case studies (e.g., universal dependencies) show performance trade-offs and how language similarity, alignment quality, and typological features influence success. "
            "The lecture also highlights evaluation pitfalls and why consistent tag sets and treebank harmonization matter."
        ),
    },
    "dialogue-chatbots": {
        "title": "Dialogue: Chatbots",
        "summary": (
            "Introduces open-domain conversational agents from rule-based systems to neural seq2seq models. "
            "It contrasts retrieval-based chatbots that pick responses from a fixed index with generative transformers that produce free-form replies. "
            "Topics include handling context windows, avoiding dull responses, incorporating persona memory, and safety considerations such as toxicity filtering. "
            "Industry examples and benchmarks like ConvAI and BlenderBot illustrate how research ideas translate to production bots."
        ),
    },
    "ethics-bias": {
        "title": "Ethics: Bias",
        "summary": (
            "Defines algorithmic bias in NLP, showing how models can amplify disparities present in training data. "
            "Students learn fairness metrics (demographic parity, equalized odds), measurement tools like bias benchmarks, and mitigation tactics including data balancing, adversarial training, and calibrated decision thresholds. "
            "The lecture revisits examples such as biased resume screening and sentiment systems that penalize African American English, stressing the human impact of deployment. "
            "It frames bias mitigation as an ongoing process requiring auditing, stakeholder engagement, and transparency."
        ),
    },
    "ethics-dangers-of-automation": {
        "title": "Ethics: Dangers of Automation",
        "summary": (
            "Discusses risks from wide-scale NLP automation: labor displacement, over-reliance on imperfect systems, and amplified misinformation. "
            "Examples include automated moderation errors, hallucinated medical advice, and the societal costs of replacing human interpreters or customer agents without robust oversight. "
            "The video encourages risk assessments, human-in-the-loop designs, and domain-specific safety evaluations before deployment, referencing failures that prompted regulatory scrutiny. "
            "Students leave with a checklist for assessing downstream harm beyond accuracy metrics."
        ),
    },
    "ethics-exclusion": {
        "title": "Ethics: Exclusion",
        "summary": (
            "Highlights how NLP systems often ignore or underperform on marginalized languages, dialects, and user groups. "
            "Topics include data availability gaps, inequities in annotation labor, and UI barriers that exclude users with disabilities. "
            "Case studies cover speech systems that fail on accents, keyboards lacking indigenous scripts, and translation models missing critical medical terminology. "
            "The lecture proposes inclusive dataset sourcing, participatory design, and evaluation across demographic slices as concrete steps toward equitable NLP."
        ),
    },
    "ethics-unethical-use-and-paths-forward": {
        "title": "Ethics: Unethical Use and Paths Forward",
        "summary": (
            "Explores malicious applications of NLP such as mass surveillance, deepfake text generation for propaganda, and automated harassment. "
            "The instructor surveys policy responses, red-teaming practices, capability assessments, and model cards as means to surface limitations. "
            "Students examine governance proposals (e.g., AI incident databases, licensing regimes) and how research labs are adopting Responsible AI frameworks. "
            "The session ends with guidance on ethical escalation channels and community norms for responsible publication."
        ),
    },
    "extractive-summarization": {
        "title": "Extractive Summarization",
        "summary": (
            "Covers algorithms that create summaries by selecting salient sentences from the source document. "
            "Classical approaches like TF-IDF scoring, maximal marginal relevance, and graph-based methods (TextRank, LexRank) are compared to modern neural extractors with sentence encoders and pointer networks. "
            "The lecture discusses evaluation with ROUGE, coverage versus redundancy trade-offs, and how extractive summaries support downstream tasks such as news aggregation. "
            "It also notes limitations (lack of abstraction, discourse coherence) that motivate abstractive models."
        ),
    },
    "language-grounding": {
        "title": "Language Grounding",
        "summary": (
            "Examines how agents connect linguistic symbols to perception and action. "
            "Examples span instruction-following robots, embodied agents in simulators, and multimodal datasets linking language to sensorimotor traces. "
            "Students learn about grounding strategies: perceptual feature learning, aligning trajectories with textual commands, and interactive learning with human feedback. "
            "Challenges such as compositional generalization and spurious correlations are highlighted alongside promising results from recent grounding benchmarks."
        ),
    },
    "language-and-vision": {
        "title": "Language and Vision",
        "summary": (
            "Introduces multimodal models that jointly reason over images (or video) and text. "
            "Tasks include visual question answering, image captioning, referring expression comprehension, and emerging contrastive models like CLIP that align embeddings across modalities. "
            "The lecture reviews dataset design (COCO, VQA, VizWiz) and architectures combining CNN or ViT backbones with transformers. "
            "It also surfaces open issues: bias in visual datasets, grounding hallucinations, and evaluating reasoning steps."
        ),
    },
    "mt-framework-and-evaluation": {
        "title": "MT: Framework and Evaluation",
        "summary": (
            "Presents the end-to-end machine translation pipeline: preprocessing, model training, decoding, and post-editing. "
            "Evaluation metrics such as BLEU, chrF, METEOR, and human adequacy/fluency judgments are compared, including their strengths and blind spots. "
            "Students see typical production workflows (translation memories, terminology constraints) and how online A/B tests complement offline metrics. "
            "Attention is given to error analysis categories and the importance of domain adaptation."
        ),
    },
    "mt-ibm-models": {
        "title": "MT: IBM Models",
        "summary": (
            "Reviews the classical IBM Models 1-5 that underpin statistical machine translation. "
            "The lecture derives noisy-channel formulation, word-to-word translation probabilities, fertility, and distortion components, highlighting the EM algorithm for training on parallel corpora. "
            "Students work through Model 1 alignment expectations and see why more complex models capture word order and multi-word phrases. "
            "These insights lay the groundwork for phrase-based SMT and alignment heuristics still used in modern pipelines."
        ),
    },
    "mt-word-alignment": {
        "title": "MT: Word Alignment",
        "summary": (
            "Explains how alignments between source and target tokens are inferred and used in translation systems. "
            "Topics include running IBM Model 1/2 in both directions, symmetrization heuristics (grow-diag-final), and alignment visualization for error analysis. "
            "The lecture also covers alignment quality metrics, handling null alignments, and the role of alignment in lexicalized reordering and terminology extraction. "
            "Modern neural alignment approaches leveraging attention weights are contrasted with traditional statistical methods."
        ),
    },
    "machine-translation-intro": {
        "title": "Machine Translation Intro",
        "summary": (
            "Sets up the translation problem, from rule-based and statistical approaches to modern neural systems. "
            "The instructor highlights linguistic challenges (word order, morphology, idioms) and resource considerations (parallel corpora, domain adaptation). "
            "Historically important milestones like the ALPAC report, pivot language techniques, and the advent of neural MT are charted to give context. "
            "Students leave understanding why translation remains an active research area despite decades of progress."
        ),
    },
    "morphology": {
        "title": "Morphology",
        "summary": (
            "Surveys morphological phenomena (inflection, derivation, compounding) and their impact on NLP. "
            "Techniques for morphological analysis range from rule-based finite-state transducers to neural sequence-to-sequence models that learn paradigm transformations. "
            "Datasets like SIGMORPHON shared tasks and UD morphological tags provide evaluation settings. "
            "The lecture also touches on leveraging subword models, character CNNs, and multitask learning to capture morphology in downstream tasks."
        ),
    },
    "multi-hop-qa": {
        "title": "Multi-hop QA",
        "summary": (
            "Covers question answering tasks that require reasoning over multiple documents or facts, exemplified by HotpotQA and QAngaroo. "
            "Model architectures combine retrieval modules, graph-based reasoning over entity links, and iterative attention that chains evidence. "
            "Students examine failure cases like reasoning shortcuts and how supporting facts supervision mitigates them. "
            "Evaluation emphasizes both answer accuracy and evidence recall to ensure genuine multi-hop reasoning."
        ),
    },
    "neural-chatbots": {
        "title": "Neural Chatbots",
        "summary": (
            "Dives deeper into neural architectures for conversational agents, including encoder-decoder transformers, hierarchical context encoders, and reinforcement learning for dialogue policy. "
            "It compares maximum likelihood training with reinforcement learning from user feedback to optimize long-term coherence. "
            "The session also reflects on controllable generation (emotion, persona, grounding) and dataset curation challenges. "
            "Connections to large conversational models like BlenderBot and LaMDA illustrate current capabilities and limitations."
        ),
    },
    "neural-and-pre-trained-machine-translation": {
        "title": "Neural and Pre-Trained Machine Translation",
        "summary": (
            "Shows how modern MT stacks pair sequence-to-sequence transformers with large-scale pre-training. "
            "Covered topics include bilingual vs multilingual training, back-translation, knowledge distillation, and leveraging pretrained models like mBART, mT5, and GPT-style decoders. "
            "Students see empirical gains from low-resource transfer and domain adaptation, plus deployment considerations such as latency and vocabulary selection. "
            "The lecture points to research on constraint decoding and quality estimation to keep MT outputs faithful."
        ),
    },
    "open-domain-qa": {
        "title": "Open-domain QA",
        "summary": (
            "Explores systems that answer factoid queries without a fixed context paragraph. "
            "Pipelines combine retrieval (BM25, dense embeddings) with neural readers like DrQA, DPR, or Fusion-in-Decoder. "
            "The lecture compares retrieve-then-read, generate-then-verify, and fully parametric approaches, noting trade-offs in accuracy, latency, and hallucination risk. "
            "Datasets such as Natural Questions and TriviaQA are used to illustrate evaluation challenges and answer validation strategies."
        ),
    },
    "phrase-based-machine-translation": {
        "title": "Phrase-based Machine Translation",
        "summary": (
            "Describes the phrase-based SMT paradigm that replaced word-for-word translation. "
            "Core components include phrase extraction from aligned corpora, phrase tables with translation probabilities, reordering models, and log-linear decoding via beam search. "
            "Students inspect how feature functions (language model, lexical weighting, distortion penalties) are combined and tuned with minimum error rate training. "
            "Although neural MT dominates today, understanding phrase-based SMT helps interpret classic baselines and hybrid systems."
        ),
    },
    "pre-trained-summarization-and-factuality": {
        "title": "Pre-trained Summarization and Factuality",
        "summary": (
            "Focuses on abstractive summarizers built on pretrained seq2seq models like BART, PEGASUS, and T5. "
            "It reviews fine-tuning strategies, dataset considerations (CNN/DailyMail, XSum), and decoding controls to balance brevity and coverage. "
            "Factuality concerns are front and center: students learn about hallucination taxonomies, automatic metrics (QAGS, FactCC), and human evaluation protocols. "
            "Mitigation techniques such as faithfulness loss terms, constrained decoding, and retrieval-augmented summarization are discussed."
        ),
    },
    "problems-with-reading-comprehension": {
        "title": "Problems with Reading Comprehension",
        "summary": (
            "Diagnoses pitfalls in RC datasets and models, including annotation artifacts, shallow heuristics, and adversarial examples. "
            "Findings from work like Jia & Liang's adversarial SQuAD and Kaushik & Lipton's perturbation studies highlight how models exploit superficial cues. "
            "The lecture encourages rigorous evaluation: contrast sets, adversarial filtering, and probing for reasoning steps. "
            "It motivates later content on explainability and dataset curation."
        ),
    },
    "reading-comprehension-intro": {
        "title": "Reading Comprehension Intro",
        "summary": (
            "Introduces machine reading comprehension as answering questions about a passage, framing it as span extraction, multiple choice, or free-form generation. "
            "Key datasets (SQuAD, NewsQA, RACE) and task variants (extractive vs abstractive answers) are surveyed. "
            "Baseline pipelines—tokenization, embedding, attention between question and passage—set expectations for subsequent architectural deep dives. "
            "Evaluation metrics like exact match and token-level F1 are explained along with their limitations."
        ),
    },
    "reading-comprehension-setup-and-baselines": {
        "title": "Reading Comprehension: Setup and Baselines",
        "summary": (
            "Builds on the intro by detailing baseline neural models such as BiDAF, Match-LSTM, and DrQA readers. "
            "Students examine how these models encode passage and question, compute attention flows, and predict span start and end positions. "
            "Implementation tips include handling out-of-vocabulary words, leveraging character CNNs, and using distant supervision for weakly labeled data. "
            "Strong lexical baselines and information retrieval heuristics are compared to emphasize the benefit of neural attention."
        ),
    },
    "reinforcement-learning-from-human-feedback-rlhf": {
        "title": "Reinforcement Learning from Human Feedback (RLHF)",
        "summary": (
            "Details the workflow popularized by InstructGPT and ChatGPT: collect preference data, train a reward model, then fine-tune a language model with PPO or similar algorithms. "
            "The lecture discusses why supervised fine-tuning alone is insufficient, how reward hacking arises, and the role of safety constraints during sampling. "
            "Practical considerations include dataset scaling, rater guidelines, and KL penalties to keep outputs close to the pre-trained distribution. "
            "Emerging research on direct preference optimization and constitutional AI is highlighted as alternative alignment strategies."
        ),
    },
    "summarization-intro": {
        "title": "Summarization Intro",
        "summary": (
            "Provides a landscape view of text summarization, distinguishing extractive and abstractive paradigms. "
            "Use cases span news, scientific articles, meetings, and legal documents, each with different compression ratios and fidelity needs. "
            "The lecture reviews classical pipelines, evaluation metrics (ROUGE, human judgments), and challenges like content selection, discourse ordering, and factual correctness. "
            "It sets the stage for subsequent lectures on specific modeling techniques."
        ),
    },
    "task-oriented-dialogue": {
        "title": "Task-Oriented Dialogue",
        "summary": (
            "Focuses on goal-driven conversational systems that help users complete tasks (booking, support, navigation). "
            "The pipeline of NLU (intent/slot extraction), dialogue state tracking, policy optimization, and NLG is explained with datasets like MultiWOZ. "
            "Approaches range from modular pipelines to end-to-end neural models trained with supervised and reinforcement learning signals. "
            "Evaluation covers success rate, task completion time, and user satisfaction, along with techniques for incorporating APIs and knowledge bases."
        ),
    },
    "n-gram-lms": {
        "title": "n-gram LMs",
        "summary": (
            "Revisits statistical language models that estimate token probabilities from fixed-order n-gram counts. "
            "Concepts include Markov assumptions, maximum likelihood estimation, and smoothing techniques such as Laplace, Katz backoff, and Kneser-Ney. "
            "The lecture illustrates perplexity computation and how n-gram LMs underpin early speech recognition and MT systems. "
            "Limitations (data sparsity, lack of long-range context) motivate the transition to neural language models covered later."
        ),
    },
}


def slugify(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-") or "item"


def clean_summary(raw: str) -> str:
    raw = FOOTNOTE_PATTERN.sub("", raw)
    raw = DOMAIN_PATTERN.sub("", raw)
    raw = raw.replace("—", "-")
    raw = raw.replace("–", "-")
    raw = re.sub(r"cs\.utexas\.edu", "", raw, flags=re.I)
    raw = re.sub(r"(https?://)?[A-Za-z0-9.-]+\.[A-Za-z]{2,}[^\s]*", "", raw)
    raw = raw.replace("\u2019", "'")
    raw = raw.replace("\u2013", "-")
    raw = raw.replace("\u2014", "-")
    raw = raw.replace("\u00a0", " ")
    # Preserve paragraph breaks but collapse internal whitespace
    paragraphs = [
        WHITESPACE_PATTERN.sub(" ", para).strip() for para in raw.split("\n")
    ]
    paragraphs = [p for p in paragraphs if p]
    return "\n\n".join(paragraphs)


def build_sentence_summaries(summary: str) -> Tuple[str, str]:
    sentences = [
        s.strip()
        for s in re.split(r"(?<=[.!?])\s+", summary.replace("\n", " "))
        if s.strip()
    ]
    if not sentences:
        return summary, ""

    overview_sentences = sentences[:2]
    remainder = sentences[2:] or sentences
    key_sentences = remainder[:4]

    overview_text = " ".join(overview_sentences).strip()
    key_text = "\n".join(f"- {s}" for s in key_sentences)
    return overview_text, key_text


def parse_notes() -> Dict[str, Dict[str, str]]:
    entries: Dict[str, Dict[str, str]] = {}
    current_key = None
    current_title = None
    buffer: list[str] = []

    for raw_line in NOTES_PATH.read_text(encoding="utf-8").splitlines():
        line = raw_line.rstrip("\n")
        stripped = line.strip()

        if stripped.startswith("•"):
            if current_key and buffer:
                summary = clean_summary("\n".join(buffer))
                if summary:
                    entries[current_key] = {
                        "title": current_title or "",
                        "summary": summary,
                    }
            header_body = stripped.lstrip("•").strip()
            header, sep, rest = header_body.partition("–")
            if sep == "":
                current_key = None
                current_title = None
                buffer = []
                continue
            header_lower = header.lower()
            if "video" not in header_lower:
                current_key = None
                current_title = None
                buffer = []
                continue

            # Remove any parenthetical mentioning video
            cleaned_title = re.sub(
                r"\(.*?video.*?\)", "", header, flags=re.I
            ).strip()
            cleaned_title = re.sub(r"\bvideo\b", "", cleaned_title, flags=re.I).strip(" :-")
            current_title = cleaned_title or header.strip()
            current_key = slugify(current_title)
            buffer = [rest.strip()] if rest.strip() else []
        else:
            if current_key:
                if stripped.startswith("Week "):
                    summary = clean_summary("\n".join(buffer))
                    if summary:
                        entries[current_key] = {
                            "title": current_title or "",
                            "summary": summary,
                        }
                    current_key = None
                    current_title = None
                    buffer = []
                else:
                    buffer.append(line)

    if current_key and buffer:
        summary = clean_summary("\n".join(buffer))
        if summary:
            entries[current_key] = {
                "title": current_title or "",
                "summary": summary,
            }
    return entries


def update_notebook(nb_path: Path, summary: str) -> None:
    nb = nbf.read(nb_path, as_version=4)
    overview_text, key_points = build_sentence_summaries(summary)
    for cell in nb.cells:
        if cell.cell_type != "markdown":
            continue
        if cell.source.startswith("## Overview"):
            content = "## Overview\n"
            content += overview_text if overview_text else summary
            cell.source = content
        elif cell.source.startswith("## Key ideas"):
            content = "## Key ideas\n"
            content += key_points if key_points else "- " + summary
            cell.source = content
    nbf.write(nb, nb_path)


def main() -> None:
    if not NOTES_PATH.exists():
        raise SystemExit(f"Missing notes file: {NOTES_PATH}")
    notes = parse_notes()
    for key, value in FALLBACK_SUMMARIES.items():
        notes.setdefault(key, value)
    idx = json.loads(COURSE_INDEX.read_text(encoding="utf-8"))
    updated = []
    missing = []

    for week in idx.get("weeks", []):
        title = week.get("week_title", "")
        match = re.match(r"Week\s+(\d+):\s*(.*)", title)
        if match:
            folder = f"{int(match.group(1)):02d}-{slugify(match.group(2))}"
        else:
            folder = slugify(title)
        week_dir = NOTEBOOK_ROOT / folder
        if not week_dir.exists():
            continue
        for item in week.get("items", []):
            if item.get("type") != "video":
                continue
            video_title = item.get("title", "").strip()
            key = slugify(video_title)
            entry = notes.get(key)
            if not entry:
                missing.append(video_title)
                continue
            summary = entry.get("summary", "").strip()
            if not summary:
                missing.append(video_title)
                continue
            nb_path = week_dir / f"{slugify(video_title)}.ipynb"
            if not nb_path.exists():
                missing.append(video_title)
                continue
            update_notebook(nb_path, summary)
            updated.append(video_title)

    log_path = Path("enrich_missing.log")
    if missing:
        log_path.write_text("\n".join(sorted(set(missing))), encoding="utf-8")
    elif log_path.exists():
        log_path.unlink()
    print(f"Updated {len(updated)} notebooks")
    if missing:
        print(f"Missing summaries for {len(set(missing))} videos; see enrich_missing.log")


if __name__ == "__main__":
    main()
