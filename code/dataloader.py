from torch.utils.data import Dataset
import torch
import pandas as pd


class CustomGraphDataset(Dataset):
    def __init__(
        self,
        X_df: pd.DataFrame,
        Y_df: pd.DataFrame,
    ):
        self.X_df = X_df
        self.Y_df = Y_df

    def __len__(self):
        return len(self.X_df)

    def __getitem__(self, idx: int) -> tuple[
        torch.tensor,
        torch.tensor,
        torch.tensor,
        torch.tensor,
    ]:
        head = self.X_df.iloc[idx]["Head"]  # .values[0]
        tail = self.X_df.iloc[idx]["Tail"]  # .values[0]
        relation = self.X_df.iloc[idx]["Relation"]  # .values[0]
        Y = self.Y_df.iloc[idx]["label"]  # .values[0]

        return head, relation, tail, Y
