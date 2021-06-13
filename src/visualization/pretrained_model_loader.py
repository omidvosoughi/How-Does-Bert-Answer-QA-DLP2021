from enum import Enum
from typing import Tuple, List

import torch
from transformers import BertForQuestionAnswering, BertTokenizer, BertConfig
from transformers import AlbertForQuestionAnswering, AlbertTokenizer, AlbertConfig

from .data_utils import QASample, SquadExample, QAInputFeatures, RawResult, read_squad_example, \
    convert_qa_example_to_features, parse_prediction


class ModelFamily(Enum):
    BERT = 1
    ALBERT = 2
    GPT2 = 3


"""
    Model loader class for loading pretained transformer models from the transformer lib.
    Its possible to load a varity of models including:
        - Bert
        - GPT-2
        - Albert

    @Params:
        model_path
        model_type
        lower_case
        cache_dir
        device
"""
class QAModel:

    def __init__(self, model_path: str, model_type: str, model_family: ModelFamily, lower_case: bool, cache_dir: str = "../cache", device: str = "cpu"):
        self.model_path = model_path
        self.model_type = model_type
        self.model_family = model_family
        self.lower_case = lower_case
        self.cache_dir = cache_dir
        self.device = device

        self._model = self.load_model()
        self._tokenizer = self.load_tokenizer()

    def load_model(self):
        pretrained_weights = torch.load(
            self.model_path, map_location=torch.device(self.device))
        if self.model_family == ModelFamily.BERT:
            config = BertConfig.from_pretrained(
                self.model_type, output_hidden_states=True, cache_dir=self.cache_dir)
            model = BertForQuestionAnswering.from_pretrained(self.model_type,
                                                             state_dict=pretrained_weights,
                                                             config=config,
                                                             cache_dir=self.cache_dir)
        if self.model_family == ModelFamily.ALBERT:
            config = AlbertConfig.from_pretrained(
                self.model_type, output_hidden_states=True, cache_dir=self.cache_dir)
            model = AlbertForQuestionAnswering.from_pretrained(self.model_type,
                                                               state_dict=pretrained_weights,
                                                               config=config,
                                                               cache_dir=self.cache_dir)

        return model

    def load_tokenizer(self):
        if self.model_family == ModelFamily.BERT:
            return BertTokenizer.from_pretrained(self.model_type, cache_dir=self.cache_dir, do_lower_case=self.lower_case)
        if self.model_family == ModelFamily.ALBERT:
            return AlbertTokenizer.from_pretrained(self.model_type, cache_dir=self.cache_dir, do_lower_case=self.lower_case)

    def predict(self, sample: QASample):
        squad_formatted_sample: SquadExample = read_squad_example(sample)

        input_features: QAInputFeatures = self.tokenize(squad_formatted_sample)

        with torch.no_grad():
            inputs = {'input_ids': input_features.input_ids,
                      'attention_mask': input_features.input_mask,
                      'token_type_ids': input_features.segment_ids
                      }

            output: Tuple = self._model(**inputs)
            prediction, hidden_states = self.__parse_model_output(output, squad_formatted_sample, input_features)

            return prediction, hidden_states, input_features

    def tokenize(self, input_sample: SquadExample) -> QAInputFeatures:
        features = convert_qa_example_to_features(example=input_sample,
                                                  tokenizer=self._tokenizer,
                                                  max_seq_length=384,
                                                  doc_stride=128,
                                                  max_query_length=64,
                                                  is_training=False)

        features.input_ids = torch.tensor([features.input_ids], dtype=torch.long)
        features.input_mask = torch.tensor([features.input_mask], dtype=torch.long)
        features.segment_ids = torch.tensor([features.segment_ids], dtype=torch.long)
        features.cls_index = torch.tensor([features.cls_index], dtype=torch.long)
        features.p_mask = torch.tensor([features.p_mask], dtype=torch.float)

        return features

    def __parse_model_output(self, output: Tuple, sample: SquadExample, features: QAInputFeatures) -> Tuple:
        def to_list(tensor):
            return tensor.detach().cpu().tolist()

        result: RawResult = RawResult(unique_id=1,
                                      start_logits=to_list(output[0][0]),
                                      end_logits=to_list(output[1][0]))

        nbest_predictions: List = parse_prediction(sample, features, result)

        return nbest_predictions[0], output[2]

