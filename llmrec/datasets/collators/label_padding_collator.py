from dataclasses import dataclass

import numpy as np
from transformers import DataCollatorForSeq2Seq


@dataclass
class LabelPaddingCollator(DataCollatorForSeq2Seq):
    def __call__(self, features, return_tensors='pt'):
        if return_tensors is None:
            return_tensors = self.return_tensors

        if isinstance(features[0], list):
            features = sum(features, [])

        labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None

        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            if self.pad_to_multiple_of is not None:
                max_label_length = (max_label_length + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of * self.pad_to_multiple_of

            for feature in features:
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
                if isinstance(feature["labels"], list):
                    feature["labels"] = (
                        feature["labels"] + remainder if self.tokenizer.padding_side == "right" else remainder + feature["labels"]
                    )
                elif self.tokenizer.padding_side == "right":
                    feature["labels"] = np.concatenate([feature["labels"], remainder]).astype(np.int64)
                else:
                    feature["labels"] = np.concatenate([remainder, feature["labels"]]).astype(np.int64)

        return {
            "inputs": self.tokenizer.pad(
                features,
                padding=self.padding,
                max_length=self.args.model_repr_max_length + self.args.data_repr_max_length,
                return_tensors=return_tensors,
                pad_to_multiple_of=self.pad_to_multiple_of,
            )
        }
