import os
import torch
import pandas as pd
import torch.nn as nn
from Config import Config
from lightning import LightningModule
from utils import create_neg_sample
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from dataloader import CustomGraphDataset
from lightning import Trainer, seed_everything
from TranslationalDistanceModel import TranslationalDistanceModel
from TransNet import TransNet

SEED = 5
seed_everything(SEED, workers=True)
config = Config()

## Dummy dataset
dummy_graph_dict = {
    "Head": [0, 1, 2, 3, 3, 4],
    "Relation": [1, 0, 0, 1, 1, 0],
    "Tail": [1, 2, 0, 5, 4, 5],
}
dummy_graph_df = pd.DataFrame.from_dict(dummy_graph_dict)
# dummy_graph_df.head()


negative_sample_graph_df = create_neg_sample(dummy_graph_df)
dummy_graph_df["label"] = [1] * len(dummy_graph_df)
negative_sample_graph_df["label"] = [-1] * len(negative_sample_graph_df)

dataset_df = pd.concat([dummy_graph_df, negative_sample_graph_df], axis=0).reset_index(
    drop=True
)

config.node_count = len(
    list(set(list(dataset_df["Head"].unique()) + list(dataset_df["Tail"].unique())))
)
config.edge_count = len(list(dataset_df["Relation"].unique()))
# dataset_df.head(10)
print("Count node", config.node_count, "edge", config.edge_count)
X_train, X_test, y_train, y_test = train_test_split(
    dataset_df[["Head", "Relation", "Tail"]],
    dataset_df[["label"]],
    test_size=0.2,
    random_state=42,
)
## Reset the index
X_train.reset_index(drop=True, inplace=True)
X_test.reset_index(drop=True, inplace=True)
y_train.reset_index(drop=True, inplace=True)
y_test.reset_index(drop=True, inplace=True)

# print(X_train, X_test, y_train, y_test)


def main():
    train_dataset = CustomGraphDataset(X_df=X_train, Y_df=y_train)
    valid_dataset = CustomGraphDataset(X_df=X_test, Y_df=y_test)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        valid_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=True,
    )

    lit_model = TransNet(config=config)
    trainer = Trainer(
        max_epochs=config.epoch,
        accelerator="auto",
        log_every_n_steps=None,
        enable_progress_bar=True,  # Explicitly enable progress bar
        logger=False,  # Disable other loggers if only using cmd output
        # callbacks=[checkpoint_callback],
    )  # Added accelerator gpu, can be cpu also, devices set to 1

    trainer.fit(lit_model, train_loader, val_loader)


if __name__ == "__main__":
    main()
