import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import random

class NetFlowDataset(Dataset):
    def __init__(self, x, y=None):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = None if y is None else torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        if self.y is None:
            return self.x[idx]
        return self.x[idx], self.y[idx]

def create_federated_datasets(
    file_path,
    iid=True,
    labelled=True,
    num_clients=100,
    num_shards=None,
    shard_size=None,
    test_size=0.2,
    seed=42
):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    df = pd.read_csv(file_path)

    feature_cols = [
        'L4_SRC_PORT', 'L4_DST_PORT', 'PROTOCOL', 'TCP_FLAGS',
        'L7_PROTO', 'IN_BYTES', 'OUT_BYTES',
        'IN_PKTS', 'OUT_PKTS', 'FLOW_DURATION_MILLISECONDS'
    ]

    X = df[feature_cols].values
    y = df['Label'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    clients = {}

    if iid:
        idx = np.random.permutation(len(X_train))
        samples = len(X_train) // num_clients

        for c in range(num_clients):
            sel = idx[c*samples:(c+1)*samples]
            clients[c] = NetFlowDataset(
                X_train[sel],
                y_train[sel] if labelled else None
            )
    else:
        assert num_shards and shard_size

        order = np.argsort(y_train)
        Xs, ys = X_train[order], y_train[order]

        shards = [(Xs[i*shard_size:(i+1)*shard_size],
                   ys[i*shard_size:(i+1)*shard_size])
                  for i in range(num_shards)]

        random.shuffle(shards)
        shards_per_client = num_shards // num_clients

        for c in range(num_clients):
            part = shards[c*shards_per_client:(c+1)*shards_per_client]
            Xc = np.vstack([p[0] for p in part])
            yc = np.hstack([p[1] for p in part])
            clients[c] = NetFlowDataset(Xc, yc if labelled else None)

    test_dataset = NetFlowDataset(X_test, y_test)
    return clients, test_dataset
