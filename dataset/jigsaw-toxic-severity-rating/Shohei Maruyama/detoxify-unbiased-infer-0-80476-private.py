import logging
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import transformers
import pickle
import pathlib

experiment_id = "jigsaw4-detoxify-train-v07"

def main():
    set_seed(0)
    logger = get_logger()

    with open("../input/jigsaw4-detoxify-train-v07/jigsaw4-detoxify-train-v07_epoch10.pickle", mode = "rb") as fp:
        estimator = pickle.load(fp)

    comments_to_score = pd.read_csv("../input/jigsaw-toxic-severity-rating/comments_to_score.csv")
    comments_to_score = comments_to_score
    dataset_test = TestDataset(comments_to_score)
    comments_to_score["score"] = estimator.predict(dataset_test)
    comments_to_score[["comment_id", "score"]].to_csv("submission.csv", header = True, index = False)

def set_seed(seed):
    # https://pytorch.org/docs/stable/notes/randomness.html
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_logger():
    logger = logging.getLogger("main")
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter("[%(asctime)s] %(message)s"))
    logger.addHandler(handler)

    return logger

class TrainingDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __len__(self):
        return 100000

    def __getitem__(self, index):
        index = np.random.randint(0, self.data.shape[0])
        return {
            "less": self.data["less_toxic"].iloc[index],
            "more": self.data["more_toxic"].iloc[index],
            "y": torch.tensor(self.data["y"].iloc[index], dtype = torch.float, device = self.device)
        }

class TestDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return {
            "text": self.data["text"].iloc[index]
        }

class Estimator:
    def __init__(self, epochs, batch_size):
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None

    def __getstate__(self):
        state = {
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "model": self.model.state_dict()
        }

        return state

    def __setstate__(self, state):
        self.epochs = state["epochs"]
        self.batch_size = state["batch_size"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Model().to(self.device)
        self.model.load_state_dict(state["model"])

    def fit(self, dataset, callbacks_epoch = [], callbacks_step = []):
        dataloader = torch.utils.data.DataLoader(dataset, batch_size = self.batch_size, shuffle = True)
        self.model = self.__init_model()
        criterion = nn.MarginRankingLoss()
        optimizer = self.__init_optimizer(self.model)
        for epoch in range(self.epochs):
            self.model.train()
            for step, batch in enumerate(dataloader):
                optimizer.zero_grad()
                batch_score_less = self.model(batch["less"])
                batch_score_more = self.model(batch["more"])
                loss = criterion(batch_score_more, batch_score_less, batch["y"])
                loss.backward()
                optimizer.step()
                locals_snapshot = locals()
                if any([callback(epoch, step, self, locals_snapshot) for callback in callbacks_step]):
                    break
            locals_snapshot = locals()
            if any([callback(epoch, self, locals_snapshot) for callback in callbacks_epoch]):
                break

        return self

    def predict(self, dataset):
        if type(dataset) is TrainingDataset:
            return self.__predict_train(dataset)
        if type(dataset) is TestDataset:
            return self.__predict_test(dataset)

    def __predict_train(self, dataset):
        dataloader = torch.utils.data.DataLoader(dataset, batch_size = self.batch_size, shuffle = False)
        self.model.eval()
        score_less = []
        score_more = []
        for batch in dataloader:
            with torch.no_grad():
                batch_score_less = self.model(batch["less"])
                score_less.append(batch_score_less.cpu().detach().numpy())
                batch_score_more = self.model(batch["more"])
                score_more.append(batch_score_more.cpu().detach().numpy())
        score_less = np.concatenate(score_less)
        score_more = np.concatenate(score_more)

        return score_less, score_more

    def __predict_test(self, dataset):
        dataloader = torch.utils.data.DataLoader(dataset, batch_size = self.batch_size, shuffle = False)
        self.model.eval()
        score = []
        for batch in dataloader:
            with torch.no_grad():
                batch_score = self.model(batch["text"])
                score.append(batch_score.cpu().detach().numpy())
        score = np.concatenate(score)

        return score

    def __init_model(self):
        model = Model().to(self.device)
        for param in model.parameters():
            param.requires_grad = False
        for param in model.text_encoder_unbiased.encoder.layer[-2].parameters():
            param.requires_grad = True
        for param in model.text_encoder_unbiased.encoder.layer[-1].parameters():
            param.requires_grad = True
        for param in model.fc.parameters():
            param.requires_grad = True

        return model

    def __init_optimizer(self, model):
        optimizer = torch.optim.AdamW([
            {"params": model.text_encoder_unbiased.encoder.layer[-2].parameters(), "lr": 0.000250},
            {"params": model.text_encoder_unbiased.encoder.layer[-1].parameters(), "lr": 0.000500},
            {"params": model.fc.parameters(), "lr": 0.001000}
        ], eps = 1e-6, weight_decay = 1e-4)

        return optimizer

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        pretrained_model_path_unbiased = "../input/detoxify/unbiased"
        config_unbiased = transformers.RobertaConfig.from_pretrained(f"{pretrained_model_path_unbiased}/config.json")
        self.text_tokenizer_unbiased = transformers.RobertaTokenizer.from_pretrained(pretrained_model_path_unbiased)
        self.text_encoder_unbiased = transformers.RobertaForSequenceClassification.from_pretrained(
            f"{pretrained_model_path_unbiased}/pytorch_model.bin",
            config = config_unbiased
        ).roberta

        self.fc = nn.Sequential(
            nn.Dropout(),
            nn.Linear(in_features = 768, out_features = 1)
        )

    def forward(self, comment):
        tokenized_unbiased = self.text_tokenizer_unbiased(comment, padding = True, truncation = True, max_length = 256, return_tensors = "pt")
        tokenized_unbiased = { key: tokenized_unbiased[key].to(self.device) for key in tokenized_unbiased }
        encoded_unbiased = self.text_encoder_unbiased(**tokenized_unbiased)["last_hidden_state"]
        encoded_unbiased = encoded_unbiased[:, 0, :]  # cls

        x = self.fc(encoded_unbiased)
        x = torch.flatten(x)

        return x

if __name__ == "__main__":
    main()