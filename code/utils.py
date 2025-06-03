import random
import pandas as pd


def create_neg_sample(df: pd.DataFrame) -> pd.DataFrame:
    dummy_neg_graph_dict = {"Head": [], "Relation": [], "Tail": []}
    neg_sample_df = pd.DataFrame.from_dict(dummy_neg_graph_dict)
    while len(neg_sample_df) < len(df):
        head = random.choice(
            list(set(list(df["Head"].unique()) + list(df["Tail"].unique())))
        )
        tail = random.choice(
            list(set(list(df["Head"].unique()) + list(df["Tail"].unique())))
        )
        if head != tail:
            relation = random.choice(list(df["Relation"].unique()))
            if (
                len(df.query("Head==@head and Tail==@tail and Relation==@relation"))
                == 0
            ):
                ## A new edge is detected
                dummy_neg_graph_dict["Head"].append(head)
                dummy_neg_graph_dict["Tail"].append(tail)
                dummy_neg_graph_dict["Relation"].append(relation)
                neg_sample_df = pd.DataFrame.from_dict(
                    dummy_neg_graph_dict
                ).drop_duplicates()

    ## return same number of neg sample
    return neg_sample_df
