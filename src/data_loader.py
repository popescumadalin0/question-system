from datasets import DatasetDict, load_dataset


def load_squad() -> DatasetDict:
    """
    Uses HF Datasets to load SQuAD v1.
    """
    ds = load_dataset("squad")
    return ds
