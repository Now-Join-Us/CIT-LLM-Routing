import importlib

from transformers import HfArgumentParser

from modelrec.utils import set_seed
from modelrec.arguments import ModelArguments, DataArguments, TrainingArguments

def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    model_core = getattr(importlib.import_module('modelrec.models'), model_args.rec_model_class)
    model_wrapper, model_kwargs_for_trainer = model_core(
        model_args=model_args,
        training_args=training_args
    )
    set_seed(training_args.seed)

    dataset_core = getattr(importlib.import_module('modelrec.datasets'), data_args.rec_training_dataset_class)
    training_dataset, data_kwargs_for_trainer = dataset_core(
        data_args=data_args,
        tokenizer=model_wrapper.tokenizer,
        data_repr_encoders=model_wrapper.data_repr_encoders,
        model_repr_encoders=model_wrapper.model_repr_encoders
    )

    trainer_core = getattr(importlib.import_module('modelrec.trainers'), training_args.rec_trainer_class)
    trainer_wrapper = trainer_core(
        training_args=training_args,
        model_wrapper=model_wrapper,
        training_dataset=training_dataset,
        **model_kwargs_for_trainer,
        **data_kwargs_for_trainer
    )
