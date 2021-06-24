import json
from dataclasses import dataclass

from jiant.tasks.lib.templates.shared import labels_to_bimap
from jiant.tasks.lib.templates import two_span_single_label
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
class Example(two_span_single_label.Example):
    @property
    def task(self):
        return SupTask


class SupTask(two_span_single_label.AbstractProbingTask):
    Example = Example
    TokenizedExample = two_span_single_label.TokenizedExample
    DataRow = two_span_single_label.DataRow
    Batch = two_span_single_label.Batch

    LABELS = [False, True]
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
                span2 = target['span2']
                examples.append(
                    Example(
                        guid=f'{set_type}-{line["info"]["q_id"]}-{str(target_id)}',
                        text=line['text'],
                        span1=span1,
                        span2=span2,
                        label=target['label']    if set_type != "test" else cls.LABELS[-1]
                        )
                    )
        return examples
