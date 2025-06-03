import torch
import torch.nn as nn
from Config import Config
from lightning import LightningModule
from TranslationalDistanceModel import TranslationalDistanceModel


class TransNet(LightningModule):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.model = TranslationalDistanceModel(config=config)
        self.train_loss = []
        self.val_loss = []
        # self.loss_fn = nn.CrossEntropyLoss(
        #    ignore_index=self.config.Y_padding_idx  # Mask out padding positions
        # )

    def loss_fn(self, l2_norm, Y):
        ## TODO : Check the dimentionality
        d = -1 * Y * l2_norm
        a = torch.log(1 + torch.exp(d))
        loss = torch.mean(a)
        return loss

    def forward(self, head, relation, tail):
        l2_norm = self.model(head=head, relation=relation, tail=tail)
        return l2_norm

    def training_step(self, batch):

        head, relation, tail, Y = batch

        l2_norm = self(head, relation, tail)  # (batch, tgt_len, vocab_size)

        loss = self.loss_fn(l2_norm, Y)
        self.train_loss.append(loss.view(1).cpu())
        return loss

    def validation_step(self, batch):

        head, relation, tail, Y = batch

        l2_norm = self(head, relation, tail)  # (batch, tgt_len, vocab_size)

        loss = self.loss_fn(l2_norm, Y)
        self.val_loss.append(loss.view(1).cpu())
        return loss

    def on_train_epoch_end(self):
        # Calculate epoch accuracy

        if len(self.train_loss) > 0:
            # print("train log")
            self.log(
                "train_loss",
                torch.cat(self.train_loss).mean(),
                prog_bar=True,
                # on_epoch=True,
            )
        # Reset lists
        self.train_loss = []

    def on_validation_epoch_end(self):
        # Calculate epoch accuracy

        if len(self.val_loss) > 0:
            # print("val log")
            self.log(
                "val_loss",
                torch.cat(self.val_loss).mean(),
                prog_bar=True,
                # on_epoch=True,
            )
        # Reset lists
        self.val_loss = []

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.LR)

        return {
            "optimizer": optimizer,
        }
