# coding=utf-8
#
# This file contains code from The Google AI Language Team Authors and The HuggingFace Inc. team, which
# was partly changed and extended in order to benefit the analysis of BERT's layers in fine-tuned QA tasks.
#
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from .sample_wrapper import QASample
import math
import collections
import unicodedata
from transformers.tokenization_utils import _is_control, _is_punctuation, _is_whitespace

logger = logging.getLogger(__name__)


def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens


class BasicTokenizer(object):
    """
    Constructs a BasicTokenizer that will run basic tokenization (punctuation splitting, lower casing, etc.).

    Args:
        do_lower_case (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not to lowercase the input when tokenizing.
        never_split (:obj:`Iterable`, `optional`):
            Collection of tokens which will never be split during tokenization. Only has an effect when
            :obj:`do_basic_tokenize=True`
        tokenize_chinese_chars (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not to tokenize Chinese characters.

            This should likely be deactivated for Japanese (see this `issue
            <https://github.com/huggingface/transformers/issues/328>`__).
        strip_accents: (:obj:`bool`, `optional`):
            Whether or not to strip all accents. If this option is not specified, then it will be determined by the
            value for :obj:`lowercase` (as in the original BERT).
    """

    def __init__(
        self,
        do_lower_case=True,
        never_split=None,
        tokenize_chinese_chars=True,
        strip_accents=None,
    ):
        if never_split is None:
            never_split = []
        self.do_lower_case = do_lower_case
        self.never_split = set(never_split)
        self.tokenize_chinese_chars = tokenize_chinese_chars
        self.strip_accents = strip_accents

    def tokenize(self, text, never_split=None):
        """
        Basic Tokenization of a piece of text. Split on "white spaces" only, for sub-word tokenization, see
        WordPieceTokenizer.

        Args:
            **never_split**: (`optional`) list of str
                Kept for backward compatibility purposes. Now implemented directly at the base class level (see
                :func:`PreTrainedTokenizer.tokenize`) List of token not to split.
        """
        # union() returns a new set by concatenating the two sets.
        never_split = (
            self.never_split.union(set(never_split))
            if never_split
            else self.never_split
        )
        text = self._clean_text(text)

        # This was added on November 1st, 2018 for the multilingual and Chinese
        # models. This is also applied to the English models now, but it doesn't
        # matter since the English models were not trained on any Chinese data
        # and generally don't have any Chinese data in them (there are Chinese
        # characters in the vocabulary because Wikipedia does have some Chinese
        # words in the English Wikipedia.).
        if self.tokenize_chinese_chars:
            text = self._tokenize_chinese_chars(text)
        orig_tokens = whitespace_tokenize(text)
        split_tokens = []
        for token in orig_tokens:
            if token not in never_split:
                if self.do_lower_case:
                    token = token.lower()
                    if self.strip_accents is not False:
                        token = self._run_strip_accents(token)
                elif self.strip_accents:
                    token = self._run_strip_accents(token)
            split_tokens.extend(self._run_split_on_punc(token, never_split))

        output_tokens = whitespace_tokenize(" ".join(split_tokens))
        return output_tokens

    def _run_strip_accents(self, text):
        """Strips accents from a piece of text."""
        text = unicodedata.normalize("NFD", text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            if cat == "Mn":
                continue
            output.append(char)
        return "".join(output)

    def _run_split_on_punc(self, text, never_split=None):
        """Splits punctuation on a piece of text."""
        if never_split is not None and text in never_split:
            return [text]
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if _is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1

        return ["".join(x) for x in output]

    def _tokenize_chinese_chars(self, text):
        """Adds whitespace around any CJK character."""
        output = []
        for char in text:
            cp = ord(char)
            if self._is_chinese_char(cp):
                output.append(" ")
                output.append(char)
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

    def _is_chinese_char(self, cp):
        """Checks whether CP is the codepoint of a CJK character."""
        # This defines a "chinese character" as anything in the CJK Unicode block:
        #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        #
        # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
        # despite its name. The modern Korean Hangul alphabet is a different block,
        # as is Japanese Hiragana and Katakana. Those alphabets are used to write
        # space-separated words, so they are not treated specially and handled
        # like the all of the other languages.
        if (
            (cp >= 0x4E00 and cp <= 0x9FFF)
            or (cp >= 0x3400 and cp <= 0x4DBF)  #
            or (cp >= 0x20000 and cp <= 0x2A6DF)  #
            or (cp >= 0x2A700 and cp <= 0x2B73F)  #
            or (cp >= 0x2B740 and cp <= 0x2B81F)  #
            or (cp >= 0x2B820 and cp <= 0x2CEAF)  #
            or (cp >= 0xF900 and cp <= 0xFAFF)
            or (cp >= 0x2F800 and cp <= 0x2FA1F)  #
        ):  #
            return True

        return False

    def _clean_text(self, text):
        """Performs invalid character removal and whitespace cleanup on text."""
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xFFFD or _is_control(char):
                continue
            if _is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)


class SquadExample(object):
    """
    A single training/test example for the Squad dataset.
    For examples without an answer, the start and end position are -1.
    """

    def __init__(
        self,
        qas_id,
        question_text,
        doc_tokens,
        sup_ids=None,
        orig_answer_text=None,
        start_position=None,
        end_position=None,
        is_impossible=None,
    ):
        self.qas_id = qas_id
        self.question_text = question_text
        self.doc_tokens = doc_tokens
        self.sup_ids = sup_ids
        self.orig_answer_text = orig_answer_text
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "qas_id: %s" % (self.qas_id)
        s += ", question_text: %s" % (self.question_text)
        s += ", doc_tokens: [%s]" % (" ".join(self.doc_tokens))
        if self.sup_ids:
            s += ", sup_ids: %d" % (self.sup_ids)
        if self.start_position:
            s += ", start_position: %d" % (self.start_position)
        if self.end_position:
            s += ", end_position: %d" % (self.end_position)
        if self.is_impossible:
            s += ", is_impossible: %r" % (self.is_impossible)
        return s


class QAInputFeatures(object):
    """A single set of features of data."""

    def __init__(
        self,
        unique_id,
        example_index,
        doc_span_index,
        tokens,
        token_to_orig_map,
        input_ids,
        input_mask,
        decoder_ids,
        segment_ids,
        cls_index,
        p_mask,
        paragraph_len,
        sup_ids=None,
        start_position=None,
        end_position=None,
        is_impossible=None,
    ):
        self.unique_id = unique_id
        self.example_index = example_index
        self.doc_span_index = doc_span_index
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map
        self.input_ids = input_ids
        self.decoder_ids = decoder_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.cls_index = cls_index
        self.p_mask = p_mask
        self.paragraph_len = paragraph_len
        self.sup_ids = sup_ids
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible


class TextInputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, tokens):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.tokens = tokens


def read_squad_example(example: QASample):
    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    paragraph_text = example.context
    doc_tokens = []
    char_to_word_offset = []
    prev_is_whitespace = True
    for c in paragraph_text:
        if is_whitespace(c):
            prev_is_whitespace = True
        else:
            if prev_is_whitespace:
                doc_tokens.append(c)
            else:
                doc_tokens[-1] += c
            prev_is_whitespace = False
        char_to_word_offset.append(len(doc_tokens) - 1)

    qas_id = example.sample_id
    question_text = example.question
    sup_ids = example.sup_ids
    sup_token_pos_ids = []

    answer = example.answer_dict
    orig_answer_text = answer["text"]
    answer_offset = answer["answer_start"]
    answer_length = len(orig_answer_text)
    start_position = char_to_word_offset[answer_offset]
    end_position = char_to_word_offset[answer_offset + answer_length - 1]

    if sup_ids:
        for sup in sup_ids:
            sup_start_position = char_to_word_offset[sup[0]]
            sup_end_position = char_to_word_offset[sup[1] - 1]

            sup_token_pos_ids.append((sup_start_position, sup_end_position))

    # Only add answers where the text can be exactly recovered from the
    # document. If this CAN'T happen it's likely due to weird Unicode
    # stuff so we will just skip the example.
    #
    # Note that this means for training mode, every example is NOT
    # guaranteed to be preserved.
    actual_text = " ".join(doc_tokens[start_position : (end_position + 1)])
    cleaned_answer_text = " ".join(whitespace_tokenize(orig_answer_text))
    if actual_text.find(cleaned_answer_text) == -1:
        logger.warning(
            "Could not find answer: '%s' vs. '%s'", actual_text, cleaned_answer_text
        )

    return SquadExample(
        qas_id=qas_id,
        question_text=question_text,
        doc_tokens=doc_tokens,
        orig_answer_text=orig_answer_text,
        start_position=start_position,
        end_position=end_position,
        sup_ids=sup_token_pos_ids,
    )


def convert_qa_example_to_features(
    example,
    tokenizer,
    max_seq_length,
    doc_stride,
    max_query_length,
    is_training,
    cls_token_at_end=False,
    cls_token="[CLS]",
    sep_token="[SEP]",
    pad_token=0,
    sequence_a_segment_id=0,
    sequence_b_segment_id=1,
    cls_token_segment_id=0,
    pad_token_segment_id=0,
    mask_padding_with_zero=True,
):
    """Loads a data file into a list of `InputBatch`s."""

    unique_id = 1000000000

    query_tokens = tokenizer.tokenize(example.question_text)

    if len(query_tokens) > max_query_length:
        query_tokens = query_tokens[0:max_query_length]

    tok_to_orig_index = []
    orig_to_tok_index = []
    all_doc_tokens = []
    for (i, token) in enumerate(example.doc_tokens):
        orig_to_tok_index.append(len(all_doc_tokens))
        sub_tokens = tokenizer.tokenize(token)
        for sub_token in sub_tokens:
            tok_to_orig_index.append(i)
            all_doc_tokens.append(sub_token)

    # Get token position for answer span
    tok_start_position = orig_to_tok_index[example.start_position]
    if example.end_position < len(example.doc_tokens) - 1:
        tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
    else:
        tok_end_position = len(all_doc_tokens) - 1
    (tok_start_position, tok_end_position) = _improve_answer_span(
        all_doc_tokens,
        tok_start_position,
        tok_end_position,
        tokenizer,
        example.orig_answer_text,
    )

    # The -3 accounts for [CLS], [SEP] and [SEP]
    max_tokens_for_doc = max_seq_length - len(query_tokens) - 3

    # We can have documents that are longer than the maximum sequence length.
    # To deal with this we do a sliding window approach, where we take chunks
    # of the up to our max length with a stride of `doc_stride`.
    _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
        "DocSpan", ["start", "length"]
    )
    doc_spans = []
    start_offset = 0
    while start_offset < len(all_doc_tokens):
        length = len(all_doc_tokens) - start_offset
        if length > max_tokens_for_doc:
            length = max_tokens_for_doc
        doc_spans.append(_DocSpan(start=start_offset, length=length))
        if start_offset + length == len(all_doc_tokens):
            break
        start_offset += min(length, doc_stride)

    for (doc_span_index, doc_span) in enumerate(doc_spans):
        tokens = []
        token_to_orig_map = {}
        segment_ids = []

        # p_mask: mask with 1 for token than cannot be in the answer (0 for token which can be in an answer)
        # Original TF implem also keep the classification token (set to 0) (not sure why...)
        p_mask = []

        # CLS token at the beginning
        if not cls_token_at_end:
            tokens.append(cls_token)
            segment_ids.append(cls_token_segment_id)
            p_mask.append(0)
            cls_index = 0

        # Query
        for token in query_tokens:
            tokens.append(token)
            segment_ids.append(sequence_a_segment_id)
            p_mask.append(1)

        # SEP token
        tokens.append(sep_token)
        segment_ids.append(sequence_a_segment_id)
        p_mask.append(1)

        # Paragraph
        for i in range(doc_span.length):
            split_token_index = doc_span.start + i
            token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]

            tokens.append(all_doc_tokens[split_token_index])
            segment_ids.append(sequence_b_segment_id)
            p_mask.append(0)
        paragraph_len = doc_span.length

        # SEP token
        tokens.append(sep_token)
        segment_ids.append(sequence_b_segment_id)
        p_mask.append(1)

        # CLS token at the end
        if cls_token_at_end:
            tokens.append(cls_token)
            segment_ids.append(cls_token_segment_id)
            p_mask.append(0)
            cls_index = len(tokens) - 1  # Index of classification token

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(pad_token)
            input_mask.append(0 if mask_padding_with_zero else 1)
            segment_ids.append(pad_token_segment_id)
            p_mask.append(1)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        span_is_impossible = example.is_impossible
        start_position = None
        end_position = None
        sup_ids = []

        if not span_is_impossible:
            # For training, if our document chunk does not contain an annotation
            # we throw it out, since there is nothing to predict.
            doc_start = doc_span.start
            doc_end = doc_span.start + doc_span.length - 1
            out_of_span = False
            if not (tok_start_position >= doc_start and tok_end_position <= doc_end):
                out_of_span = True
            if out_of_span:
                start_position = 0
                end_position = 0
                span_is_impossible = True
            else:
                doc_offset = len(query_tokens) + 2
                start_position = tok_start_position - doc_start + doc_offset
                end_position = tok_end_position - doc_start + doc_offset

                # Get token position for supporting fact spans
                for sup in example.sup_ids:
                    sup_tok_start_position = (
                        orig_to_tok_index[sup[0]] - doc_start + doc_offset
                    )
                    sup_tok_end_position = (
                        orig_to_tok_index[sup[1]] - doc_start + doc_offset
                    )

                    sup_ids.append((sup_tok_start_position, sup_tok_end_position))

        if span_is_impossible:
            start_position = cls_index
            end_position = cls_index

        logger.info("*** Example ***")
        logger.info("unique_id: %s" % (unique_id))
        logger.info("doc_span_index: %s" % (doc_span_index))
        logger.info("tokens: %s" % " ".join(tokens))
        logger.info(
            "token_to_orig_map: %s"
            % " ".join(["%d:%d" % (x, y) for (x, y) in token_to_orig_map.items()])
        )
        logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        if span_is_impossible:
            logger.info("impossible example")
        if not span_is_impossible:
            answer_text = " ".join(tokens[start_position : (end_position + 1)])
            logger.info("start_position: %d" % (start_position))
            logger.info("end_position: %d" % (end_position))
            logger.info("answer: %s" % (answer_text))

        decoder_ids = tokenizer.convert_tokens_to_ids(
            tokens[start_position : (end_position + 1)]
        )

        return QAInputFeatures(
            unique_id=unique_id,
            example_index=0,
            doc_span_index=doc_span_index,
            tokens=tokens,
            token_to_orig_map=token_to_orig_map,
            input_ids=input_ids,
            decoder_ids=decoder_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            cls_index=cls_index,
            p_mask=p_mask,
            paragraph_len=paragraph_len,
            start_position=start_position,
            end_position=end_position,
            sup_ids=sup_ids,
            is_impossible=span_is_impossible,
        )


def convert_text_example_to_features(example, max_seq_length, tokenizer):
    """Loads a data file into an `InputBatch`."""

    tokens = tokenizer.tokenize(example.text)

    # Account for [CLS] and [SEP] with "- 2"
    if len(tokens) > max_seq_length - 2:
        tokens = tokens[: (max_seq_length - 2)]

    tokens = ["[CLS]"] + tokens + ["[SEP]"]
    segment_ids = [0] * len(tokens)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    padding = [0] * (max_seq_length - len(input_ids))
    input_ids += padding
    input_mask += padding
    segment_ids += padding

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    return TextInputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        tokens=tokens,
    )


def _improve_answer_span(
    doc_tokens, input_start, input_end, tokenizer, orig_answer_text
):
    """Returns tokenized answer spans that better match the annotated answer."""

    # The SQuAD annotations are character based. We first project them to
    # whitespace-tokenized words. But then after WordPiece tokenization, we can
    # often find a "better match". For example:
    #
    #   Question: What year was John Smith born?
    #   Context: The leader was John Smith (1895-1943).
    #   Answer: 1895
    #
    # The original whitespace-tokenized answer will be "(1895-1943).". However
    # after tokenization, our tokens will be "( 1895 - 1943 ) .". So we can match
    # the exact answer, 1895.
    #
    # However, this is not always possible. Consider the following:
    #
    #   Question: What country is the top exporter of electornics?
    #   Context: The Japanese electronics industry is the lagest in the world.
    #   Answer: Japan
    #
    # In this case, the annotator chose "Japan" as a character sub-span of
    # the word "Japanese". Since our WordPiece tokenizer does not split
    # "Japanese", we just use "Japanese" as the annotation. This is fairly rare
    # in SQuAD, but does happen.
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start : (new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)


RawResult = collections.namedtuple(
    "RawResult", ["unique_id", "start_logits", "end_logits"]
)


def parse_prediction(
    example, features, result, max_answer_length=30, do_lower_case=True, n_best_size=20
):
    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction", ["start_index", "end_index", "start_logit", "end_logit"]
    )

    prelim_predictions = []

    start_indexes = _get_best_indexes(result.start_logits, n_best_size)
    end_indexes = _get_best_indexes(result.end_logits, n_best_size)

    for start_index in start_indexes:
        for end_index in end_indexes:
            # We could hypothetically create invalid predictions, e.g., predict
            # that the start of the span is in the question. We throw out all
            # invalid predictions.
            if start_index >= len(features.tokens):
                continue
            if end_index >= len(features.tokens):
                continue
            if start_index not in features.token_to_orig_map:
                continue
            if end_index not in features.token_to_orig_map:
                continue
            if end_index < start_index:
                continue
            length = end_index - start_index + 1
            if length > max_answer_length:
                continue
            prelim_predictions.append(
                _PrelimPrediction(
                    start_index=start_index,
                    end_index=end_index,
                    start_logit=result.start_logits[start_index],
                    end_logit=result.end_logits[end_index],
                )
            )

    prelim_predictions = sorted(
        prelim_predictions, key=lambda x: (x.start_logit + x.end_logit), reverse=True
    )

    _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "NbestPrediction",
        ["text", "start_index", "end_index", "start_logit", "end_logit"],
    )

    seen_predictions = {}
    nbest = []
    for pred in prelim_predictions:
        if len(nbest) >= n_best_size:
            break

        if pred.start_index > 0:  # this is a non-null prediction
            tok_tokens = features.tokens[pred.start_index : (pred.end_index + 1)]
            orig_doc_start = features.token_to_orig_map[pred.start_index]
            orig_doc_end = features.token_to_orig_map[pred.end_index]
            orig_tokens = example.doc_tokens[orig_doc_start : (orig_doc_end + 1)]
            tok_text = " ".join(tok_tokens)

            # De-tokenize WordPieces that have been split off.
            tok_text = tok_text.replace(" ##", "")
            tok_text = tok_text.replace("##", "")

            # Clean whitespace
            tok_text = tok_text.strip()
            tok_text = " ".join(tok_text.split())
            orig_text = " ".join(orig_tokens)

            final_text = get_final_text(tok_text, orig_text, do_lower_case, True)
            if final_text in seen_predictions:
                continue

            seen_predictions[final_text] = True
        else:
            final_text = ""
            seen_predictions[final_text] = True

        nbest.append(
            _NbestPrediction(
                text=final_text,
                start_index=pred.start_index,
                end_index=pred.end_index,
                start_logit=pred.start_logit,
                end_logit=pred.end_logit,
            )
        )

    # In very rare edge cases we could only have single null prediction.
    # So we just create a nonce prediction in this case to avoid failure.
    if len(nbest) == 1:
        nbest.insert(
            0,
            _NbestPrediction(
                text="empty",
                start_logit=0.0,
                end_logit=0.0,
                start_index=-1,
                end_index=-1,
            ),
        )

    # In very rare edge cases we could have no valid predictions. So we
    # just create a nonce prediction in this case to avoid failure.
    if not nbest:
        nbest.append(
            _NbestPrediction(
                text="empty",
                start_logit=0.0,
                end_logit=0.0,
                start_index=-1,
                end_index=-1,
            )
        )

    assert len(nbest) >= 1

    total_scores = []
    best_non_null_entry = None
    for entry in nbest:
        total_scores.append(entry.start_logit + entry.end_logit)
        if not best_non_null_entry:
            if entry.text:
                best_non_null_entry = entry

    probs = _compute_softmax(total_scores)

    nbest_json = []
    for (i, entry) in enumerate(nbest):
        output = collections.OrderedDict()
        output["text"] = entry.text
        output["probability"] = probs[i]
        output["start_logit"] = entry.start_logit
        output["end_logit"] = entry.end_logit
        output["start_index"] = entry.start_index
        output["end_index"] = entry.end_index
        nbest_json.append(output)

    return nbest_json


def get_final_text(pred_text, orig_text, do_lower_case, verbose_logging=False):
    """Project the tokenized prediction back to the original text."""

    # When we created the data, we kept track of the alignment between original
    # (whitespace tokenized) tokens and our WordPiece tokenized tokens. So
    # now `orig_text` contains the span of our original text corresponding to the
    # span that we predicted.
    #
    # However, `orig_text` may contain extra characters that we don't want in
    # our prediction.
    #
    # For example, let's say:
    #   pred_text = steve smith
    #   orig_text = Steve Smith's
    #
    # We don't want to return `orig_text` because it contains the extra "'s".
    #
    # We don't want to return `pred_text` because it's already been normalized
    # (the SQuAD eval script also does punctuation stripping/lower casing but
    # our tokenizer does additional normalization like stripping accent
    # characters).
    #
    # What we really want to return is "Steve Smith".
    #
    # Therefore, we have to apply a semi-complicated alignment heuristic between
    # `pred_text` and `orig_text` to get a character-to-character alignment. This
    # can fail in certain cases in which case we just return `orig_text`.

    def _strip_spaces(text):
        ns_chars = []
        ns_to_s_map = collections.OrderedDict()
        for (i, c) in enumerate(text):
            if c == " ":
                continue
            ns_to_s_map[len(ns_chars)] = i
            ns_chars.append(c)
        ns_text = "".join(ns_chars)
        return (ns_text, ns_to_s_map)

    # We first tokenize `orig_text`, strip whitespace from the result
    # and `pred_text`, and check if they are the same length. If they are
    # NOT the same length, the heuristic has failed. If they are the same
    # length, we assume the characters are one-to-one aligned.
    tokenizer = BasicTokenizer(do_lower_case=do_lower_case)

    tok_text = " ".join(tokenizer.tokenize(orig_text))

    start_position = tok_text.find(pred_text)
    if start_position == -1:
        if verbose_logging:
            logger.info("Unable to find text: '%s' in '%s'" % (pred_text, orig_text))
        return orig_text
    end_position = start_position + len(pred_text) - 1

    (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
    (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

    if len(orig_ns_text) != len(tok_ns_text):
        if verbose_logging:
            logger.info(
                "Length not equal after stripping spaces: '%s' vs '%s'",
                orig_ns_text,
                tok_ns_text,
            )
        return orig_text

    # We then project the characters in `pred_text` back to `orig_text` using
    # the character-to-character alignment.
    tok_s_to_ns_map = {}
    for (i, tok_index) in tok_ns_to_s_map.items():
        tok_s_to_ns_map[tok_index] = i

    orig_start_position = None
    if start_position in tok_s_to_ns_map:
        ns_start_position = tok_s_to_ns_map[start_position]
        if ns_start_position in orig_ns_to_s_map:
            orig_start_position = orig_ns_to_s_map[ns_start_position]

    if orig_start_position is None:
        if verbose_logging:
            logger.info("Couldn't map start position")
        return orig_text

    orig_end_position = None
    if end_position in tok_s_to_ns_map:
        ns_end_position = tok_s_to_ns_map[end_position]
        if ns_end_position in orig_ns_to_s_map:
            orig_end_position = orig_ns_to_s_map[ns_end_position]

    if orig_end_position is None:
        if verbose_logging:
            logger.info("Couldn't map end position")
        return orig_text

    output_text = orig_text[orig_start_position : (orig_end_position + 1)]
    return output_text


def _get_best_indexes(logits, n_best_size):
    """Get the n-best logits from a list."""
    index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes


def _compute_softmax(scores):
    """Compute softmax probability over raw logits."""
    if not scores:
        return []

    max_score = None
    for score in scores:
        if max_score is None or score > max_score:
            max_score = score

    exp_scores = []
    total_sum = 0.0
    for score in scores:
        x = math.exp(score - max_score)
        exp_scores.append(x)
        total_sum += x

    probs = []
    for score in exp_scores:
        probs.append(score / total_sum)
    return probs
