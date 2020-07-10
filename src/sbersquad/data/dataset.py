import json
from pathlib import Path
import os

import torch
from deeppavlov.core.data.utils import download_decompress
from torch.utils.data import TensorDataset
from transformers import AutoTokenizer

from .utils import (read_squad_examples, convert_examples_to_features)

URL_SBERSQUAD = "http://files.deeppavlov.ai/datasets/sber_squad-v1.1.tar.gz"


def load_and_cache_examples(
        model_name_or_path: str,
        tokenizer,
        evaluate=False,
        output_examples=False,
        model_type='bert',
        max_seq_length=384,
        overwrite_cache: bool = False,
        version_2_with_negative: bool = False,
        rank=0,
):
    if rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Load data features from cache or dataset file
    input_file = "data/dev-v1.1.json" if evaluate else "data/train-v1.1.json"
    cached_features_file = os.path.join(os.path.dirname(input_file), 'cached_{}_{}_{}'.format(
        'dev' if evaluate else 'train',
        list(filter(None, model_name_or_path.split('/'))).pop(),
        str(max_seq_length)))
    if os.path.exists(cached_features_file) and not overwrite_cache and not output_examples:
        features = torch.load(cached_features_file)
    else:
        examples = read_squad_examples(input_file=input_file,
                                                is_training=not evaluate,
                                                version_2_with_negative=version_2_with_negative)
        features = convert_examples_to_features(examples=examples,
                                                tokenizer=tokenizer,
                                                cls_token='<BOS>' if model_type in ['roberta', 'ruberta'] else '[CLS]',
                                                sep_token='<EOS>' if model_type in ['roberta', 'ruberta'] else '[SEP]',
                                                pad_token=1 if model_type in ['roberta', 'ruberta'] else 0,
                                                add_prefix_space=True if model_type == 'roberta' else False,
                                                max_seq_length=max_seq_length,
                                                doc_stride=128,
                                                max_query_length=64,
                                                is_training=not evaluate)
        if rank in [-1, 0]:
            torch.save(features, cached_features_file)

    if rank >= 1 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_cls_index = torch.tensor([f.cls_index for f in features], dtype=torch.long)
    all_p_mask = torch.tensor([f.p_mask for f in features], dtype=torch.float)
    if evaluate:
        all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                all_example_index, all_cls_index, all_p_mask)
    else:
        all_start_positions = torch.tensor([f.start_position for f in features], dtype=torch.long)
        all_end_positions = torch.tensor([f.end_position for f in features], dtype=torch.long)
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                all_start_positions, all_end_positions,
                                all_cls_index, all_p_mask)

    if output_examples:
        return dataset, examples, features
    return dataset


class SbersquadDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            path: str,
            model_name:str,
            tokenizer: AutoTokenizer = None,
            url: str = None,
            train: bool = True
    ):
        self.url = url or URL_SBERSQUAD
        self.path = path
        dir_path = Path(path)
        required_files = ['{}-v1.1.json'.format(dt) for dt in ['train', 'dev']]
        if not dir_path.exists():
            dir_path.mkdir()

        if not all((dir_path / f).exists() for f in required_files):
            download_decompress(self.url, dir_path)
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        if train:
            self.dataset = load_and_cache_examples(
                model_name_or_path=model_name,
                tokenizer=tokenizer
            )
        else:
            self.dataset = load_and_cache_examples(
                model_name_or_path=model_name,
                tokenizer=tokenizer,
                evaluate=True
            )

    def __getitem__(self, idx):
        out = self.dataset[idx]
        out_dict = {
            "input_ids": out[0],
            "attention_mask": out[1],
            "token_type_ids": out[2],
            "start_positions": out[3],
            "end_positions":   out[4]
        }
        return out_dict

    def __len__(self):
        return len(self.dataset)
