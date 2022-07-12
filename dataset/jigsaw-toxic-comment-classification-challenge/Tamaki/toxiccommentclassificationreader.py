import csv
from typing import Dict
from overrides import overrides

import numpy as np

from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField, MetadataField, ArrayField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer


@DatasetReader.register("toxic_comment")
class ToxicCommentClassificationReader(DatasetReader):
    def __init__(self,
                 lazy: bool = False,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 max_sequence_length: int = None) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._max_sequence_length = max_sequence_length

    @overrides
    def _read(self, file_path):
        with open(file_path, "r") as data_file:
            csv_in = csv.reader(data_file)
            # skip header
            next(csv_in)
            for row in csv_in:
                if len(row) == 8:
                    # train
                    yield self.text_to_instance(
                        comment_text=row[1],
                        labels=np.array(row[2:])
                    )
                else:
                    # test
                    yield self.text_to_instance(
                        comment_text=row[1],
                        labels=None
                    )

    def _truncate(self, tokens):
        if len(tokens) > self._max_sequence_length:
            tokens = tokens[:self._max_sequence_length]
        return tokens

    @overrides
    def text_to_instance(self, comment_text: str,
                         labels: np.array = None) -> Instance:
        tokenized_comment = self._tokenizer.tokenize(comment_text)
        if len(tokenized_comment) == 0:
            tokenized_comment = [Token('__NULL__')]
        if self._max_sequence_length is not None:
            tokenized_comment = self._truncate(tokenized_comment)
        comment_field = TextField(tokenized_comment, self._token_indexers)
        fields = {'tokens': comment_field}

        if labels is not None:
            array_field = ArrayField(array=labels)
            fields["label"] = array_field

        return Instance(fields)
