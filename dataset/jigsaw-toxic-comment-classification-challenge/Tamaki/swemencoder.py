from typing import Dict

from overrides import overrides
import torch

from allennlp.common.checks import ConfigurationError
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import Seq2SeqEncoder, Seq2VecEncoder, TextFieldEmbedder
from allennlp.nn import InitializerApplicator
from allennlp.nn.util import get_text_field_mask
from allennlp.nn.util import get_lengths_from_binary_sequence_mask


class SWEMEncoder(Seq2VecEncoder):
    POOL_TYPES = ['avg', 'max', 'concat']
    def __init__(self,
                 embedding_dim: int,
                 pool_type: str = 'concat'):
        super(SWEMEncoder, self).__init__()
        self._embedding_dim = embedding_dim

        if not pool_type in self.POOL_TYPES:
            raise ConfigurationError("")

        self._pool_type = pool_type

    @overrides
    def get_input_dim(self) -> int:
        return self._embedding_dim

    @overrides
    def get_output_dim(self) -> int:
        if self._pool_type == 'concat':
            return self._embedding_dim * 2
        return self._embedding_dim

    def _average_pooling(self, tokens, mask):
        summed = tokens.sum(1)

        if mask is not None:
            lengths = get_lengths_from_binary_sequence_mask(mask)
            length_mask = (lengths > 0)

            # Set any length 0 to 1, to avoid dividing by zero.
            lengths = torch.max(lengths, lengths.new_ones(1))
        else:
            lengths = tokens.new_full((1,), fill_value=tokens.size(1))
            length_mask = None

        avg_pool = summed / lengths.unsqueeze(-1).float()

        if length_mask is not None:
            avg_pool = summed * (length_mask > 0).float().unsqueeze(-1)

        return avg_pool

    def _max_pooling(self, tokens):
        return tokens.max(1)[0]

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor = None):  #pylint: disable=arguments-differ
        if mask is not None:
            tokens = tokens * mask.unsqueeze(-1).float()

        pool = None

        if self._pool_type == 'avg':
            pool = self._average_pooling(tokens, mask)
        elif self._pool_type == 'max':
            pool = self._max_pooling(tokens)
        else:
            max_pool = self._max_pooling(tokens)
            avg_pool = self._average_pooling(tokens, mask)
            pool = torch.cat((max_pool, avg_pool), 1)

        return pool