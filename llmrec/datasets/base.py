from torch.utils.data import Dataset

import random
from arguments import DataArguments
from typing import Any, Tuple, List, Optional

from modelrec.representations import DataRepresentationEncoder, ModelRepresentationEncoder
from modelrec.utils import load_json

from transformers import PreTrainedTokenizer
from functools import reduce

class RecDataset(Dataset):
    def __init__(
        self,
        data_args: DataArguments,
        tokenizer: PreTrainedTokenizer,
        data_repr_encoders: Optional[List[DataRepresentationEncoder]] = None,
        model_repr_encoders: Optional[List[ModelRepresentationEncoder]] = None,
    ):
        self.dataset = sum([load_json(data_file) for data_file in data_args.rec_training_data_url], [])
        random.shuffle(self.dataset)
        self.tokenizer = tokenizer
        self.data_repr_encoders = [] if data_repr_encoders is None else data_repr_encoders
        self.model_repr_encoders = [] if model_repr_encoders is None else model_repr_encoders
        self.data_args = data_args

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item, apply_repr_encode=True) -> Tuple[Any, Any]:
        data_repr = self.dataset[item]['data_repr']
        model_repr = self.dataset[item]['model_repr']

        if apply_repr_encode:
            data_repr = reduce(lambda x, func: func.apply(x), self.data_repr_encoders, data_repr)
            model_repr = reduce(lambda x, func: func.apply(x), self.model_repr_encoders, model_repr)

        return (data_repr, model_repr)
