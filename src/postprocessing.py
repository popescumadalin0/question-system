from typing import Dict
import evaluate

import numpy as np

def postprocess_qa_predictions(
        examples,
        features,
        raw_predictions,
        n_best_size: int = 20,
        max_answer_length: int = 30,
):
    """
    Converts start/end logits into final text answers.
    Based on HF QA examples (simplified).
    """
    all_start_logits, all_end_logits = raw_predictions
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = {}

    for i, feature in enumerate(features):
        example_index = example_id_to_index[feature["example_id"]]
        features_per_example.setdefault(example_index, []).append(i)

    predictions = {}

    for example_index, example in enumerate(examples):
        feature_indices = features_per_example.get(example_index, [])
        context = example["context"]

        valid_answers = []
        for feature_index in feature_indices:
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            offset_mapping = features[feature_index]["offset_mapping"]

            start_indexes = np.argsort(start_logits)[-1:-n_best_size - 1:-1].tolist()
            end_indexes = np.argsort(end_logits)[-1:-n_best_size - 1:-1].tolist()

            for start_index in start_indexes:
                for end_index in end_indexes:
                    if start_index >= len(offset_mapping) or end_index >= len(offset_mapping):
                        continue
                    if offset_mapping[start_index] is None or offset_mapping[end_index] is None:
                        continue
                    if end_index < start_index:
                        continue

                    start_char = offset_mapping[start_index][0]
                    end_char = offset_mapping[end_index][1]
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue

                    answer_text = context[start_char:end_char]
                    score = start_logits[start_index] + end_logits[end_index]
                    valid_answers.append({"score": float(score), "text": answer_text})

        if len(valid_answers) > 0:
            best = max(valid_answers, key=lambda x: x["score"])
            predictions[example["id"]] = best["text"]
        else:
            predictions[example["id"]] = ""

    return predictions


def compute_metrics_squad(examples, predictions: Dict[str, str]) -> Dict[str, float]:
    """
    Uses evaluate's SQuAD metric (Exact Match & F1).
    """
    metric = evaluate.load("squad")
    formatted_preds = [{"id": k, "prediction_text": v} for k, v in predictions.items()]
    references = [{"id": ex["id"], "answers": ex["answers"]} for ex in examples]
    return metric.compute(predictions=formatted_preds, references=references)
