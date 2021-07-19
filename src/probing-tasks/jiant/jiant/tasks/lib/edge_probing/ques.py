"""Question Type Classification Edge Probing task.

"""
from dataclasses import dataclass

from jiant.tasks.lib.templates.shared import labels_to_bimap
from jiant.tasks.lib.templates import edge_probing_single_span
from jiant.utils.python.io import read_json_lines


@dataclass
class Example(edge_probing_single_span.Example):
    @property
    def task(self):
        return QuesTask


@dataclass
class TokenizedExample(edge_probing_single_span.TokenizedExample):
    pass


@dataclass
class DataRow(edge_probing_single_span.DataRow):
    pass


@dataclass
class Batch(edge_probing_single_span.Batch):
    pass


class QuesTask(edge_probing_single_span.AbstractProbingTask):
    Example = Example
    TokenizedExample = TokenizedExample
    DataRow = DataRow
    Batch = Batch

    # Labels adopted from https://cogcomp.seas.upenn.edu/Data/QA/QC/definition.html
    LABELS = [
        "ABBR:abb", "ABBR:exp",
        "ENTY:animal", "ENTY:body", "ENTY:color", "ENTY:cremat", "ENTY:currency", "ENTY:dismed",
        "ENTY:event", "ENTY:food", "ENTY:instrument", "ENTY:lang", "ENTY:letter", "ENTY:other",
        "ENTY:plant", "ENTY:product", "ENTY:religion", "ENTY:sport", "ENTY:substance", "ENTY:symbol",
        "ENTY:techmeth", "ENTY:termeq", "ENTY:veh", "ENTY:word", 
        "DESC:def", "DESC:desc", "DESC:manner", "DESC:reason", "DESC:yesno",
        "HUM:gr", "HUM:ind", "HUM:title", "HUM:desc", 
        "LOC:city", "LOC:country", "LOC:mount", "LOC:other", "LOC:state", 
        "NUM:code", "NUM:count", "NUM:date", "NUM:dist", "NUM:money", "NUM:ord", "NUM:other",
        "NUM:period", "NUM:perc", "NUM:speed", "NUM:temp", "NUM:volsize", "NUM:weight",
    ]
    LABEL_TO_ID, ID_TO_LABEL = labels_to_bimap(LABELS)

    @property
    def num_spans(self):
        return 1

    def get_train_examples(self):
        return self._create_examples(lines=read_json_lines(self.train_path), set_type="train")

    def get_val_examples(self):
        return self._create_examples(lines=read_json_lines(self.val_path), set_type="val")

    def get_test_examples(self):
        return self._create_examples(lines=read_json_lines(self.test_path), set_type="test")

    @classmethod
    def _create_examples(cls, lines, set_type):
        examples = []
        for (line_num, line) in enumerate(lines):
            for (target_num, target) in enumerate(line["targets"]):
                span = target["span1"]
                examples.append(
                    Example(
                        guid="%s-%s-%s" % (set_type, line_num, target_num),
                        text=line["text"],
                        span=span,
                        labels=[target["label"]] if set_type != "test" else [cls.LABELS[-1]],
                    )
                )
        return examples