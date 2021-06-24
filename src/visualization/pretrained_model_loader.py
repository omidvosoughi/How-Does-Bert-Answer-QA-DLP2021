from enum import Enum
from typing import Tuple, List

import torch
# from transformers import BertForQuestionAnswering, BertTokenizer, BertConfig
# from transformers import AlbertForQuestionAnswering, AlbertTokenizer, AlbertConfig

from transformers import AutoTokenizer, AutoModelForQuestionAnswering, AutoConfig, PreTrainedModel
from .data_utils import QASample, SquadExample, QAInputFeatures, RawResult, read_squad_example, \
    convert_qa_example_to_features, parse_prediction


class ModelType(Enum):
    BERT_BASE_UNCASED = "bert-base-uncased"
    BERT_BASE_UNCASED_SQUAD = "csarron/bert-base-uncased-squad-v1"
    T5_BASE_QA_QG = "valhalla/t5-base-qa-qg-hl"


"""
    Model loader class for loading pretained transformer models from the transformer lib or custom transformers.
    @Params:
        model_path
        custom
        model_type
        cache_dir
        device
"""


class QAModel:

    def __init__(self, head: PreTrainedModel, model_type: ModelType=None, custom=False, model_path=None, cache_dir: str = "../cache", device: str = "cpu"):
        self.model_type = model_type
        self.model_path = model_path
        self.cache_dir = cache_dir
        self.device = device
        self.head = head

        self._model = self.load_model(custom)
        self._tokenizer = self.load_tokenizer()

    def load_model(self, custom):
        config = AutoConfig.from_pretrained(
                self.model_type.value, output_hidden_states=True, cache_dir=self.cache_dir)
        if custom:
            pretrained_weights = torch.load(
                self.model_path, map_location=torch.device(self.device))
            model = self.head.from_pretrained(
                self.model_path, state_dict=pretrained_weights, config=config, cache_dir=self.cache_dir)
        else:
            model = self.head.from_pretrained(
                self.model_type.value, config=config, cache_dir=self.cache_dir)
        return model

    def load_tokenizer(self):
        return AutoTokenizer.from_pretrained(self.model_type.value)

    def predict(self, sample: QASample):
        squad_formatted_sample: SquadExample = read_squad_example(sample)
        input_features: QAInputFeatures = self.tokenize(squad_formatted_sample)

        with torch.no_grad():
            inputs = {'input_ids': input_features.input_ids,
                      'attention_mask': input_features.input_mask,
                      'token_type_ids': input_features.segment_ids
                      }

            """
            inputs = {'input_ids': input_features.input_ids,
                      'attention_mask': input_features.input_mask,
                      'decoder_input_ids': input_features.decoder_ids
                      }
            """

            output: Tuple = self._model(**inputs)
            print(output, output[0][0].shape, output[1][0].shape)
            prediction, hidden_states = self.__parse_model_output(
                output, squad_formatted_sample, input_features)

            return prediction, hidden_states, input_features

    def tokenize(self, input_sample: SquadExample) -> QAInputFeatures:
        features = convert_qa_example_to_features(example=input_sample,
                                                  tokenizer=self._tokenizer,
                                                  max_seq_length=384,
                                                  doc_stride=128,
                                                  max_query_length=64,
                                                  is_training=False)

        print(features)

        features.input_ids = torch.tensor(
            [features.input_ids], dtype=torch.long)
        features.decoder_ids = torch.tensor(
            [features.decoder_ids], dtype=torch.long)
        features.input_mask = torch.tensor(
            [features.input_mask], dtype=torch.long)
        features.segment_ids = torch.tensor(
            [features.segment_ids], dtype=torch.long)
        features.cls_index = torch.tensor(
            [features.cls_index], dtype=torch.long)
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
