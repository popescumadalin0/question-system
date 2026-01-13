import spacy
from transformers import Trainer, DefaultDataCollator, AutoModelForQuestionAnswering, AutoTokenizer, \
    set_seed, TrainingArguments

from data_loader import load_squad
from postprocessing import compute_metrics_squad, postprocess_qa_predictions
from preprocessing import prepare_validation_features, prepare_train_features
from qaconfig import QAConfig
from utilities import spacy_preprocess_context, basic_text_clean, ensure_nltk


# -----------------------------
# Model train/eval
# -----------------------------
def train_and_evaluate(args):
    ensure_nltk()
    nlp = spacy.load("en_core_web_sm")

    set_seed(args.seed)

    print("Loading dataset (SQuAD)...")
    ds = load_squad()

    # Light preprocessing with spaCy (context only) - optional but meets requirement
    def spacy_map(batch):
        batch["context"] = [spacy_preprocess_context(nlp, c) for c in batch["context"]]
        batch["question"] = [basic_text_clean(q) for q in batch["question"]]
        return batch

    if args.use_spacy_clean:
        ds = ds.map(spacy_map, batched=True)

    cfg = QAConfig(
        model_name=args.model_name,
        max_length=args.max_length,
        doc_stride=args.doc_stride,
        pad_on_right=True,
    )

    print(f"Loading tokenizer/model: {cfg.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
    model = AutoModelForQuestionAnswering.from_pretrained(cfg.model_name)

    print("Tokenizing train/validation...")
    train_features = ds["train"].map(
        lambda x: prepare_train_features(x, tokenizer, cfg),
        batched=True,
        remove_columns=ds["train"].column_names,
    )

    val_features = ds["validation"].map(
        lambda x: prepare_validation_features(x, tokenizer, cfg),
        batched=True,
        remove_columns=ds["validation"].column_names,
    )

    data_collator = DefaultDataCollator()

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=args.train_bs,
        per_device_eval_batch_size=args.eval_bs,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        logging_steps=50,
        save_total_limit=2,
        fp16=args.fp16,
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_features,
        eval_dataset=val_features,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    print("Training...")
    trainer.train()

    print("Predicting on validation for SQuAD metrics...")
    raw = trainer.predict(val_features)
    # raw.predictions = (start_logits, end_logits)
    start_logits, end_logits = raw.predictions

    # Need original examples + features with offset mapping and example ids
    # Recreate val_features with needed columns by tokenizing without removing columns
    val_features_full = ds["validation"].map(
        lambda x: prepare_validation_features(x, tokenizer, cfg),
        batched=True,
    )

    preds = postprocess_qa_predictions(
        examples=ds["validation"],
        features=val_features_full,
        raw_predictions=(start_logits, end_logits),
    )
    metrics = compute_metrics_squad(ds["validation"], preds)

    print("SQuAD metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    # Save final model
    print("Saving model...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print(f"Done. Model saved to: {args.output_dir}")

