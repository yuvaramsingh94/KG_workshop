class Config:
    def __init__(self):
        self.node_count = 10
        self.edge_count = 10
        self.node_embedding_size = 32
        self.edge_embedding_size = 32
        self.model = "TransE"
        self.batch_size = 2
        self.LR = 0.0001
        self.epoch = 5
