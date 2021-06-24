import json
from dataclasses import dataclass

from jiant.tasks.lib.templates.shared import labels_to_bimap
from jiant.tasks.lib.templates import single_span_single_label
from jiant.utils.python.io import read_json_lines

from jiant.tasks.core import (
    BaseExample,
    BaseTokenizedExample,
    BaseDataRow,
    BatchMixin,
    Task,
    TaskTypes,
    )


@dataclass
class Example(single_span_single_label.Example):
    @property
    def task(self):
        return TrecTask


class TrecTask(single_span_single_label.AbstractProbingTask):
    Example = Example
    TokenizedExample = single_span_single_label.TokenizedExample
    DataRow = single_span_single_label.DataRow
    Batch = single_span_single_label.Batch
    
    LABELS = [
              "LOC:state",
              "ABBR:abb",
              "NUM:dist",
              "ENTY:cremat",
              "ENTY:termeq",
              "NUM:money",
              "DESC:desc",
              "ENTY:lang",
              "ENTY:sport",
              "ENTY:animal",
              "NUM:perc",
              "NUM:weight",
              "ABBR:exp",
              "LOC:mount",
              "ENTY:body",
              "LOC:city",
              "ENTY:color",
              "ENTY:instru",
              "NUM:speed",
              "ENTY:dismed",
              "ENTY:food",
              "NUM:other",
              "HUM:ind",
              "NUM:temp",
              "ENTY:letter",
              "ENTY:product",
              "ENTY:word",
              "ENTY:substance",
              "NUM:ord",
              "HUM:title",
              "ENTY:veh",
              "ENTY:event",
              "ENTY:religion",
              "DESC:reason",
              "DESC:def",
              "LOC:other",
              "NUM:count",
              "HUM:gr",
              "ENTY:other",
              "HUM:desc",
              "NUM:volsize",
              "ENTY:symbol",
              "LOC:country",
              "DESC:manner",
              "ENTY:currency",
              "NUM:date",
              "ENTY:plant",
              "NUM:code",
              "NUM:period",
              "ENTY:techmeth"]
    
    LABEL_TO_ID, ID_TO_LABEL = labels_to_bimap(LABELS)
    
    def get_train_examples(self):
        return self._create_examples(lines=read_json_lines(self.train_path), set_type="train")
    
    def get_val_examples(self):
        return self._create_examples(lines=read_json_lines(self.val_path), set_type="val")
    
    def get_test_examples(self):
        return self._create_examples(lines=read_json_lines(self.test_path), set_type="test")
    
    @classmethod
    def _create_examples(cls, lines, set_type):
        examples = []
        for line in lines:
            for target_id, target in enumerate(line['targets']):
                span1 = target['span1']
                examples.append(
                    Example(
                        guid=f'{set_type}-{line["info"]["q_id"]}-{str(target_id)}',
                        text=line['text'],
                        span1=span1,
                        label=target['label']    if set_type != "test" else cls.LABELS[-1]
                        )
                    )
        return examples