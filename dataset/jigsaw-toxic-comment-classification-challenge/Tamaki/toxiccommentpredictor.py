from typing import Iterable

import numpy as np
import torch
from tqdm import tqdm

from allennlp.data.instance import Instance
from allennlp.data.iterators import DataIterator
from allennlp.models.model import Model
from allennlp.nn import util as nn_util
 
class ToxicCommentPredictor:
    def __init__(self, model: Model, iterator: DataIterator,
                 cuda_device: int=-1) -> None:
        self.model = model
        self.iterator = iterator
        self.cuda_device = cuda_device
         
    def _extract_data(self, batch) -> np.ndarray:
        out_dict = self.model(**batch)
        return out_dict["probs"].detach().cpu().numpy()
     
    def predict(self, ds: Iterable[Instance]) -> np.ndarray:
        pred_generator = self.iterator(ds, num_epochs=1, shuffle=False)
        self.model.eval()
        pred_generator_tqdm = tqdm(pred_generator,
                                   total=self.iterator.get_num_batches(ds))
        preds = []
        with torch.no_grad():
            for batch in pred_generator_tqdm:
                batch = nn_util.move_to_device(batch, self.cuda_device)
                preds.append(self._extract_data(batch))
        return np.concatenate(preds, axis=0)