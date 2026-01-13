from dataclasses import dataclass

@dataclass
class QAConfig:
    model_name: str = "bert-base-uncased"
    max_length: int = 384
    doc_stride: int = 128
    pad_on_right: bool = True