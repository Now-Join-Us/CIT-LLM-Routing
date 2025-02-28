import os
from dataclasses import dataclass, field
from typing import Optional, List

from transformers import TrainingArguments

from modelrec.configs import DATA_PATH

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )

    peft_model_path: str = field(
        default=''
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    use_lora: bool = field(
        default=True,
        metadata={"help": "If passed, will use LORA (low-rank parameter-efficient training) to train the model."}
    )
    lora_rank: int = field(
        default=64,
        metadata={"help": "The rank of lora."}
    )
    lora_alpha: float = field(
        default=16,
        metadata={"help": "The alpha parameter of lora."}
    )
    lora_dropout: float = field(
        default=0.1,
        metadata={"help": "The dropout rate of lora modules."}
    )
    target_modules: List[str] = field(
        default_factory=["q_proj", "v_proj", "o_proj", "down_proj", "up_proj", "gate_proj"]
    )
    use_flash_attn: bool = field(
        default=True,
        metadata={"help": "If passed, will use flash attention to train the model."}
    )
    cache_dir: str = field(
        default="tmp", metadata={"help": "the cache of the model"}
    )
    token: str = field(
        default=None, metadata={"help": "the token to access the huggingface model"}
    )
    peft_model_path: str = field(
        default=None
    )
    modules_to_save: str = field(
        default=None
    )

@dataclass
class DataArguments:
    rec_training_data_url: str = field(
        default='toy_finetune_data.jsonl', metadata={"help": "Path to train data"}
    )

    contrastive_train_group_size: int = field(default=8)

    query_max_len: int = field(
        default=32,
        metadata={
            "help": "The maximum total input sequence length after tokenization for passage. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )

    passage_max_len: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization for passage. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )

    max_example_num_per_dataset: int = field(
        default=100000000, metadata={"help": "the max number of examples for each dataset"}
    )

    query_instruction_for_retrieval: str = field(
        default="A: ", metadata={"help": "query: "}
    )

    passage_instruction_for_retrieval: str = field(
        default="B: ", metadata={"help": "passage: "}
    )

    cache_path: str = field(
        default='./data_dir'
    )

    load_from_disk: bool = field(
        default=False, metadata={"help": " whether load the data from disk"}
    )

    load_disk_path: str = field(
        default=None, metadata={"help": " the path to load the data", "nargs": "+"}
    )

    save_to_disk: bool = field(
        default=False, metadata={"help": " whether save the data to disk"}
    )

    save_disk_path: str = field(
        default=None, metadata={"help": " the path to save the data"}
    )

    num_shards: int = field(
        default=0, metadata={
            "help": "number of shards to write, prior than `save_max_shard_size`, default depends on `save_max_shard_size`"}
    )

    save_max_shard_size: str = field(
        default="50GB", metadata={"help": "the max size of the shard"}
    )

    exit_after_save: bool = field(
        default=False, metadata={"help": " whether exit after save the data"}
    )

    def __post_init__(self):
        if not os.path.exists(self.train_data):
            raise FileNotFoundError(f"cannot find file: {self.train_data}, please set a true path")

@dataclass
class RecArguments:
    model_repr_prompt_template: str = field(
        default='toy_finetune_data.jsonl', metadata={"help": "Path to train data"}
    )

    contrastive_train_group_size: int = field(default=8)

@dataclass
class RecTrainingArguments(TrainingArguments):
    loss_type: str = field(default='only logits')

def preprocess_args(args):
    args.rec_training_data_url = args.rec_training_data_url.split(',')
    if DATA_PATH is not None:
        args.rec_training_data_url = [os.path.join(DATA_PATH, i) for i in args.rec_training_data_url]
