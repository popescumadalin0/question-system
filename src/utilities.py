import argparse
import re

import nltk

import spacy
from nltk.tokenize import sent_tokenize
from transformers import (
    pipeline,
)


# Optional sklearn (extra reporting)

def ensure_nltk():
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")


def basic_text_clean(text: str) -> str:
    """Light normalization - keep it minimal for extractive QA."""
    text = re.sub(r"\s+", " ", text).strip()
    return text


def spacy_preprocess_context(nlp, context: str) -> str:
    """
    Example preprocessing with spaCy:
    - normalize whitespace
    - optionally remove weird control chars
    We do NOT lemmatize/remove stopwords because extractive QA needs exact spans.
    """
    context = basic_text_clean(context)
    doc = nlp(context)
    # Remove non-printable chars but keep original tokens largely intact
    cleaned = "".join(ch for ch in doc.text if ch.isprintable())
    return cleaned


def answer_from_passage(args):
    ensure_nltk()
    nlp = spacy.load("en_core_web_sm")

    context = basic_text_clean(args.context)
    question = basic_text_clean(args.question)

    if args.use_spacy_clean:
        context = spacy_preprocess_context(nlp, context)

    # Optional NLTK sentence segmentation for debugging / passage chunking
    sents = sent_tokenize(context)
    if args.debug:
        print("Context sentences (NLTK):")
        for i, s in enumerate(sents[:10]):
            print(f"{i + 1}. {s}")

    qa = pipeline(
        "question-answering",
        model=args.model_path_or_name,
        tokenizer=args.model_path_or_name,
    )

    result = qa(question=question, context=context)
    print("\nAnswer:")
    print(f"  {result['answer']}")
    print(f"Score: {result['score']:.4f}")
    print(f"Span:  start={result['start']} end={result['end']}")


def build_argparser():
    p = argparse.ArgumentParser(description="Extractive QA system (BERT) with standard NLP pipeline.")
    sub = p.add_subparsers(dest="mode", required=True)

    # Train/eval on SQuAD
    t = sub.add_parser("train", help="Train and evaluate on SQuAD.")
    t.add_argument("--model_name", type=str, default="bert-base-uncased")
    t.add_argument("--output_dir", type=str, default="./qa_bert_out")
    t.add_argument("--epochs", type=float, default=1.0)
    t.add_argument("--lr", type=float, default=3e-5)
    t.add_argument("--weight_decay", type=float, default=0.01)
    t.add_argument("--train_bs", type=int, default=8)
    t.add_argument("--eval_bs", type=int, default=8)
    t.add_argument("--max_length", type=int, default=384)
    t.add_argument("--doc_stride", type=int, default=128)
    t.add_argument("--fp16", action="store_true")
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--use_spacy_clean", action="store_true", help="Apply light spaCy cleaning on contexts.")

    # Inference
    i = sub.add_parser("infer", help="Run inference on a custom (question, context).")
    i.add_argument("--model_path_or_name", type=str, default="distilbert-base-uncased-distilled-squad")
    i.add_argument("--question", type=str, required=True)
    i.add_argument("--context", type=str, required=True)
    i.add_argument("--use_spacy_clean", action="store_true")
    i.add_argument("--debug", action="store_true")

    return p
