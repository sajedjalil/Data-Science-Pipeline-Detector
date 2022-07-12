import os

# print(os.listdir("../input"))
# print(os.listdir("../input/train"))

# Any results you write to the current directory are saved as output.
import torch
import numpy as np
import pandas as pd
from torch import nn
from torch.utils.data import Dataset


class ToTensor:
    def __init__(self, excluded_keys=("length", "id")):
        if not isinstance(excluded_keys, set):
            excluded_keys = set(excluded_keys)
        self.excluded = excluded_keys

    def __call__(self, x):
        result = {k: torch.from_numpy(v) for k, v in x.items() if k not in self.excluded}
        for k in self.excluded:
            if k in x.keys():
                result[k] = x[k]
        return result


class PetDataset(Dataset):
    def __init__(self, path, class_mapping, transform=None, top=None, inference=False):
        self.transform = transform
        self.class_mapping = class_mapping
        data = pd.read_csv(path)
        self.inference = inference
        if top:
            data = data.head(top)
        if not inference:
            self.y = data["AdoptionSpeed"].astype(int)

        # list of preprocessing steps
        data.drop('Name', axis=1, inplace=True)
        data.drop('Breed2', axis=1, inplace=True)
        data.drop('State', axis=1, inplace=True)
        data.drop('RescuerID', axis=1, inplace=True)
        data.drop('Description', axis=1, inplace=True)
        data.drop('PetID', axis=1, inplace=True)
        if not inference:
            data.drop('AdoptionSpeed', axis=1, inplace=True)

        data["Type"] = (data["Type"] == 1).astype(int)  # 1 = Dog, 2 = Cat  meaning it will be 0 for cat and 1 for dog
        data['Age'] = data['Age'].apply(lambda x: np.log2(x) if x > 0 else 0)
        data['Quantity'] = data['Quantity'].apply(lambda x: np.log2(x) if x > 0 else 0)

        data["Color1"] = data["Color1"].apply(lambda x: x - 1)  # to start categories from 0

        # COLOR2 and 3 starts from 0
        data.loc[data['Gender'] == 3, 'Gender'] = 0
        data.loc[data['Sterilized'] == 3, 'Sterilized'] = 0
        data.loc[data['Sterilized'] == 2, 'Sterilized'] = -1
        data.loc[data['Dewormed'] == 3, 'Dewormed'] = 0
        data.loc[data['Dewormed'] == 2, 'Dewormed'] = -1
        data.loc[data['Vaccinated'] == 3, 'Vaccinated'] = 0
        data.loc[data['Vaccinated'] == 2, 'Vaccinated'] = -1

        data['Fee'] = data['Fee'].apply(lambda x: np.log10(x) if x > 0 else 0)
        data['PhotoAmt'] = data['PhotoAmt'].apply(lambda x: np.log2(x) if x > 0 else 0)
        data['VideoAmt'] = data['VideoAmt'].apply(lambda x: np.log2(x) if x > 0 else 0)

        data["Age"] = data["Age"].apply(lambda x: (x - 1.7945) / 1.0982 if x != 0 else 0)
        data["VideoAmt"] = data["VideoAmt"].apply(lambda x: (x - 1.3659) / 0.5053 if x != 0 else 0)
        data["PhotoAmt"] = data["PhotoAmt"].apply(lambda x: (x - 1.3659) / 0.5053 if x != 0 else 0)
        data["Fee"] = data["Fee"].apply(lambda x: (x - 1.9482) / 0.4455 if x != 0 else 0)
        data["Quantity"] = data["Quantity"].apply(lambda x: (x - 0.3743) / 0.7585 if x != 0 else 0)
        # divided by max value
        data["Gender"] = data["Gender"].apply(lambda x: x / 2)
        data["MaturitySize"] = data["MaturitySize"].apply(lambda x: x / 4)
        data["FurLength"] = data["FurLength"].apply(lambda x: x / 3)
        data["Health"] = data["Health"].apply(lambda x: x / 3)

        cat_numerical = data[
            ["Breed1", "Color1", "Color2", "Color3", "Type", "Age", "Gender", "MaturitySize", "FurLength", "Vaccinated",
             "Dewormed", "Sterilized", "Health", "Quantity", "Fee", "VideoAmt", "PhotoAmt"]]
        self.x = cat_numerical.values

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        x = self.x[idx]
        item = {"inputs": x}
        if not self.inference:
            y = np.array([self.y[idx]])  # since it is the only value need to convert to list
            item["targets"] = y
        # x = np.expand_dims(x, axis=0)
        if self.transform:
            item = self.transform(item)
        return item


class PetClassifier(nn.Module):
    def __init__(self, emb_dims, num_numeric, num_classes):
        """
        emb_dims: List of two element tuples
          This list will contain a two element tuple for each
          categorical feature. The first element of a tuple will
          denote the number of unique values of the categorical
          feature. The second element will denote the embedding
          dimension to be used for that feature.

        """
        super().__init__()

        self.inference = False
        self.activation = nn.ELU(inplace=True)
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.softmax = nn.Softmax(dim=-1)

        self.embeddings = nn.ModuleList([nn.Embedding(x, y) for x, y in emb_dims])
        no_of_embs = sum([y for x, y in emb_dims])
        self.no_of_embs = no_of_embs
        self.emb_dims = emb_dims

        self.fc1 = nn.Linear(no_of_embs + num_numeric, 10)
        self.bn1 = nn.BatchNorm1d(10)
        self.fc2 = nn.Linear(10, num_classes)

    def forward(self, x):
        # notice cast to long tensor here
        emb = [emb_layer(x[:, i].long()) for i, emb_layer in enumerate(self.embeddings)]
        emb = torch.cat(emb, 1)
        numeric = x[:, len(self.emb_dims):].float()

        data = torch.cat([emb, numeric], 1)
        out = self.fc1(data)
        out = self.activation(out)
        out = self.bn1(out)
        out = self.fc2(out)
        if self.inference:
            out = self.softmax(out)
        else:
            # using cross entropy here
            out = out
        return out


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    print(torch.__version__)

    transf = ToTensor()
    ds1 = PetDataset("../input/train/train.csv", {}, transform=transf)
    batchsize = 256
    dataloader = DataLoader(ds1, batch_size=batchsize, shuffle=True, num_workers=1, pin_memory=False, drop_last=False)
    net = PetClassifier([(308, 3), (7, 1), (8, 1), (8, 1)], 13, 5)
    no_of_epochs = 150
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adagrad(net.parameters(), lr=0.001)
    for epoch in range(no_of_epochs):
        iterations_per_epoch = len(dataloader)
        data_iterator = iter(dataloader)
        for _ in range(iterations_per_epoch):
            optimizer.zero_grad()
            next_batch = next(data_iterator)
            x, y = next_batch["inputs"], next_batch["targets"]
            y = y.squeeze(1)
            preds = net(x)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
        print(epoch, loss)

    inference_path = "../input/test/test.csv"
    predicted_labels = []
    test_pet_ids = pd.read_csv(inference_path)["PetID"]
    ds2 = PetDataset(inference_path, {}, transform=transf, inference=True)
    val_loader = DataLoader(ds2, batch_size=128)
    iterations_per_epoch = len(val_loader)
    data_iterator = iter(val_loader)
    for _ in range(iterations_per_epoch):
        next_batch = next(data_iterator)
        x = next_batch["inputs"]
        preds = net(x)
        _, classes = preds.max(-1)
        vals = classes.data.cpu().numpy().astype('int32').tolist()
        predicted_labels.extend(vals)

    submission = pd.DataFrame({'PetID': test_pet_ids, 'AdoptionSpeed': predicted_labels})
    submission.to_csv('submission.csv', index=False)