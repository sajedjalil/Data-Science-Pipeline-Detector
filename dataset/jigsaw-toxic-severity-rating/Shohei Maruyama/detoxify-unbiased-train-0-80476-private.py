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

    # 学習データ作成
    validation_data = pd.read_csv("../input/jigsaw-toxic-severity-rating/validation_data.csv")
    validation_data["less_toxic"] = validation_data["less_toxic"].str.strip()
    validation_data["more_toxic"] = validation_data["more_toxic"].str.strip()
    train = pd.read_csv("../input/jigsaw-toxic-comment-classification-challenge/train.csv")
    train["comment_text"] = train["comment_text"].str.strip()
    train["type"] = train["toxic"] * 1 + train["severe_toxic"] * 2 + train["obscene"] * 4 + train["threat"] * 8 + train["insult"] * 16 + train["identity_hate"] * 32
    type_y = (
        validation_data.
        merge(train[["comment_text", "type"]].rename(columns = {"comment_text": "less_toxic", "type": "type_less"}), how = "left", on = "less_toxic").
        merge(train[["comment_text", "type"]].rename(columns = {"comment_text": "more_toxic", "type": "type_more"}), how = "left", on = "more_toxic").
        pipe(lambda df: pd.concat([
            df.assign(y = 1),
            df.rename(columns = {"type_less": "type_more", "type_more": "type_less"}).assign(y = -1)
        ], ignore_index = True)).
        groupby(["type_less", "type_more"])["y"].agg(["mean", "count"]).
        loc[lambda df: df["mean"] > 0, :].reset_index()
    )
    augmented = []
    for index, row in type_y.iterrows():
        size = int(row["count"] * 100)
        temp = pd.DataFrame({"y": [row["mean"] for i in range(size)]})
        temp["less_toxic"] = train.loc[lambda df: df["type"] == row["type_less"], "comment_text"].sample(size, replace = True).values
        temp["more_toxic"] = train.loc[lambda df: df["type"] == row["type_more"], "comment_text"].sample(size, replace = True).values
        augmented.append(temp)
    augmented = pd.concat(augmented, ignore_index = True)
    dataset_train = TrainingDataset(augmented)

    # 検証データ作成
    dataset_val = TestDataset(
        pd.DataFrame({"text": validation_data["less_toxic"].tolist() + validation_data["more_toxic"].tolist()}).
        drop_duplicates(ignore_index = True).
        assign(num_words = lambda df: df["text"].str.split().apply(lambda x: len(x))).
        sort_values(["num_words"])
    )

    # エポック終了時点のモデルを保存するコールバック関数
    def save_snapshot(epoch, estimator, locals):
        pathlib.Path("model").mkdir(exist_ok = True)
        with open("model/{:s}_epoch{:02d}.pickle".format(experiment_id, epoch + 1), mode = "wb") as fp:
            pickle.dump(estimator, fp)
        return False

    # エポック終了後に検証データで精度を計算するコールバック関数
    def print_auc_epoch(epoch, estimator, locals):
        score = estimator.predict(dataset_val)
        score = dataset_val.data.assign(score = score)
        val_accuracy = (
            validation_data.
            merge(score.rename(columns = {"text": "less_toxic", "score": "score_less"}), how = "left", on = "less_toxic").
            merge(score.rename(columns = {"text": "more_toxic", "score": "score_more"}), how = "left", on = "more_toxic").
            assign(correct = lambda df: df["score_less"] < df["score_more"])
            ["correct"].mean()
        )
        logger.info("Epoch: {:2d}, Step:  val, Accuracy: {:6.4f}".format(epoch + 1, val_accuracy))
        return False

    # ステップ終了後に学習データのミニバッチで精度を計算するコールバック関数
    history = {}
    def print_auc_step(epoch, step, estimator, locals):
        if epoch not in history:
            history[epoch] = []
        batch_score_less = locals["batch_score_less"].cpu().detach().numpy()
        batch_score_more = locals["batch_score_more"].cpu().detach().numpy()
        batch_accuracy = (batch_score_more > batch_score_less).mean()
        history[epoch].append(batch_accuracy)
        logger.info("Epoch: {:2d}, Step: {:4d}, Accuracy: {:6.4f} ± {:6.4f}".format(epoch + 1, step + 1, np.mean(np.array(history[epoch])), np.std(np.array(history[epoch]))))
        return False

    # 学習
    estimator = Estimator(epochs = 10, batch_size = 128)
    estimator = estimator.fit(dataset_train, [save_snapshot, print_auc_epoch], [print_auc_step])

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
