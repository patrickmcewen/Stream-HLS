import torch
from torch import nn
import lightning as L
import torchmetrics
import torch.nn.functional as F
# from memory_dataset import Preprocess
from tqdm.auto import *

class DenseDecoder(nn.Module):
    def __init__(self, input_size, preprocess=None, basis=None, hidden_size=1024, mode='pretrain', augment_kwargs=None, dropouts=(0.1,0.1,0.3), delete_loss=False):
        super().__init__()
        # self.delete_loss = delete_loss
        # # self.preprocess = Preprocess(basis=basis) if preprocess is None else preprocess
        self.hidden_size = hidden_size
        self.dropouts = dropouts
        self.decoder = nn.Sequential(
            nn.Linear(input_size, self.hidden_size),
            nn.BatchNorm1d(self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropouts[0]),
            nn.Linear(self.hidden_size, self.hidden_size//2),
            nn.BatchNorm1d(self.hidden_size//2),
            nn.ReLU(),
            nn.Dropout(self.dropouts[1]),
            nn.Linear(self.hidden_size//2, self.hidden_size//4),
            nn.BatchNorm1d(self.hidden_size//4),
            nn.ReLU(),
            nn.Dropout(self.dropouts[2]),
            nn.Linear(self.hidden_size//4, 1),
            nn.Sigmoid()
        )
        # self.train_accuracy = torchmetrics.classification.Accuracy(task='binary')
        # self.sim_val_accuracy = torchmetrics.classification.Accuracy(task='binary')
        # self.exp_val_accuracy = torchmetrics.classification.Accuracy(task='binary')

        # self.basis = basis
        # self.augment_kwargs = augment_kwargs
        # self.save_hyperparameters()
        # self.mode = mode

    # def on_after_batch_transfer(self, batch, dataloader_idx):
    #     meas, det = batch
    #     return self.preprocess(meas, det, shuffle=True, apply_flips=True, has_detector_loss=False, delete_loss=self.delete_loss)

    def forward(self, X):
        return self.decoder(X)

    # def training_step(self, batch):
    #     X, y = batch
    #     pred = self.forward(X)
    #     self.train_accuracy(pred, y)
    #     self.log('train_acc', self.train_accuracy, prog_bar=True, on_step=False, on_epoch=True)
    #     loss = F.binary_cross_entropy(pred, y)
    #     self.log('train_loss', loss, prog_bar=False, on_step=False, on_epoch=True)
    #     return loss

    # def validation_step(self, batch, batch_idx=0, dataloader_idx=0):
    #     X, y = batch
    #     pred = self.forward(X)
    #     if dataloader_idx == 0:
    #         self.sim_val_accuracy(pred, y)
    #         loss = F.binary_cross_entropy(pred, y)
    #         self.log('sim_val_acc', self.sim_val_accuracy, prog_bar=True, on_step=False, on_epoch=True)
    #         self.log('sim_val_loss', loss, prog_bar=False, on_step=False, on_epoch=True)
    #     else:
    #         self.exp_val_accuracy(pred, y)
    #         loss = F.binary_cross_entropy(pred, y)
    #         self.log('exp_val_acc', self.exp_val_accuracy, prog_bar=True, on_step=False, on_epoch=True)
    #         self.log('exp_val_loss', loss, prog_bar=False, on_step=False, on_epoch=True)

    # def configure_optimizers(self):
    #     if self.mode == 'pretrain':
    #         optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3, weight_decay=1e-2)
    #         scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=10, factor=0.3, min_lr=1e-5)
    #         return {"optimizer": optimizer, "lr_scheduler": {
    #             "scheduler": scheduler,
    #             "interval": "epoch",
    #             "frequency": 1,
    #             "monitor": "exp_val_acc/dataloader_idx_1",
    #             "strict": True,
    #         }}
    #     elif self.mode == 'finetune':
    #         return torch.optim.AdamW(self.parameters(), lr=1e-3, weight_decay=8e-2)


# # test
# model = DenseDecoder(input_size=1024, basis='rbf', hidden_size=1024, mode='pretrain')
# # model.load_state_dict(torch.load('pretrain_model.pth'))
# model.eval()

# # test
# X = torch.randn(1, 1024)
# pred = model(X)
# print(pred)