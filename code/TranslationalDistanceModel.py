from Config import Config
from lightning import LightningModule
import torch.nn as nn
import torch


class TranslationalDistanceModel(LightningModule):
    def __init__(self, config: Config):
        super(TranslationalDistanceModel, self).__init__()
        self.config = config
        ## TODO Initialize the embedding layer for nodes and edges
        ### Embedding layer
        self.node_embedding = nn.Embedding(
            self.config.node_count,
            self.config.node_embedding_size,
            # padding_idx=self.config.X_padding_idx,
        )
        self.edge_embedding = nn.Embedding(
            self.config.edge_count,
            self.config.edge_embedding_size,
            # padding_idx=self.config.X_padding_idx,
        )

    def forward(self, head, relation, tail):
        # head = bs,1
        # relation = bs,1
        # tail = bs,1
        ## Embed the node and edge shape (bs,emb)
        e_head = self.node_embedding(head)
        e_tail = self.node_embedding(tail)
        e_relation = self.edge_embedding(relation)
        if self.config.model == "TransE":
            ## L2
            difference = e_head + e_relation - e_tail
            ## IP (bs,emb) -> OP (bs,1)
            l2_norm = torch.norm(difference, p="fro", dim=1)  ## Check the dimention
            ## return bs,1
            ## Negative for the scoring function
            return -l2_norm
