from qaconfig import QAConfig


def prepare_train_features(examples, tokenizer, cfg: QAConfig):
    """
    Tokenize and create start/end positions for training.
    """
    # In SQuAD: question, context, answers
    questions = [q.lstrip() for q in examples["question"]]
    contexts = examples["context"]

    tokenized = tokenizer(
        questions if cfg.pad_on_right else contexts,
        contexts if cfg.pad_on_right else questions,
        truncation="only_second" if cfg.pad_on_right else "only_first",
        max_length=cfg.max_length,
        stride=cfg.doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_mapping = tokenized.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized.pop("offset_mapping")

    start_positions = []
    end_positions = []

    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        sequence_ids = tokenized.sequence_ids(i)
        sample_index = sample_mapping[i]
        answers = examples["answers"][sample_index]

        if len(answers["answer_start"]) == 0:
            start_positions.append(cls_index)
            end_positions.append(cls_index)
            continue

        start_char = answers["answer_start"][0]
        end_char = start_char + len(answers["text"][0])

        # Find start/end token indices in the context portion
        token_start_index = 0
        while sequence_ids[token_start_index] != (1 if cfg.pad_on_right else 0):
            token_start_index += 1

        token_end_index = len(input_ids) - 1
        while sequence_ids[token_end_index] != (1 if cfg.pad_on_right else 0):
            token_end_index -= 1

        # If answer not fully inside this feature, label with CLS
        if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
            start_positions.append(cls_index)
            end_positions.append(cls_index)
        else:
            while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                token_start_index += 1
            start_positions.append(token_start_index - 1)

            while offsets[token_end_index][1] >= end_char:
                token_end_index -= 1
            end_positions.append(token_end_index + 1)

    tokenized["start_positions"] = start_positions
    tokenized["end_positions"] = end_positions
    return tokenized


def prepare_validation_features(examples, tokenizer, cfg: QAConfig):
    """
    Tokenize validation; keep offset mapping for post-processing predictions.
    """
    questions = [q.lstrip() for q in examples["question"]]
    contexts = examples["context"]

    tokenized = tokenizer(
        questions if cfg.pad_on_right else contexts,
        contexts if cfg.pad_on_right else questions,
        truncation="only_second" if cfg.pad_on_right else "only_first",
        max_length=cfg.max_length,
        stride=cfg.doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_mapping = tokenized.pop("overflow_to_sample_mapping")
    tokenized["example_id"] = []

    # Set offsets to None for question tokens so we only map to context spans
    for i in range(len(tokenized["input_ids"])):
        sequence_ids = tokenized.sequence_ids(i)
        context_index = 1 if cfg.pad_on_right else 0

        tokenized["example_id"].append(examples["id"][sample_mapping[i]])
        tokenized["offset_mapping"][i] = [
            o if sequence_ids[k] == context_index else None
            for k, o in enumerate(tokenized["offset_mapping"][i])
        ]

    return tokenized

