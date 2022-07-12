from typing import Dict, Union

from overrides import overrides
import torch
from pytorch_pretrained_bert.modeling import BertModel

from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.modules.token_embedders.bert_token_embedder import PretrainedBertModel
from allennlp.nn.initializers import InitializerApplicator
from allennlp.training.metrics import CategoricalAccuracy


@Model.register("toxic_bert")
class ToxicBertClassifier(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 bert_model: Union[str, BertModel],
                 dropout: float = 0.0,
                 num_labels: int = None,
                 index: str = "bert",
                 label_namespace: str = "labels",
                 trainable: bool = True,
                 initializer: InitializerApplicator = InitializerApplicator()) -> None:
        super().__init__(vocab)

        if isinstance(bert_model, str):
            self.bert_model = PretrainedBertModel.load(bert_model)
        else:
            self.bert_model = bert_model

        for param in self.bert_model.parameters():
            param.requires_grad = trainable

        in_features = self.bert_model.config.hidden_size

        self._label_namespace = label_namespace

        if num_labels:
            out_features = num_labels
        else:
            out_features = vocab.get_vocab_size(namespace=self._label_namespace)

        self._dropout = torch.nn.Dropout(p=dropout)

        self._classification_layer = torch.nn.Linear(in_features, out_features)
        self._loss = torch.nn.BCEWithLogitsLoss()
        self._index = index
        initializer(self._classification_layer)

    def forward(self,  # type: ignore
                tokens: Dict[str, torch.LongTensor],
                label: torch.IntTensor = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        input_ids = tokens[self._index]
        token_type_ids = tokens[f"{self._index}-type-ids"]
        input_mask = (input_ids != 0).long()

        _, pooled = self.bert_model(input_ids=input_ids,
                                    token_type_ids=token_type_ids,
                                    attention_mask=input_mask)

        pooled = self._dropout(pooled)

        # apply classification layer
        logits = self._classification_layer(pooled)

        probs = torch.nn.functional.sigmoid(logits)

        output_dict = {"logits": logits, "probs": probs}

        if label is not None:
            loss = self._loss(logits, label)
            output_dict["loss"] = loss

        return output_dict