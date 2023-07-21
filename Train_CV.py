import os
import numpy as np
from tqdm import tqdm
import scipy.sparse as sp

import torch
import torch.nn.functional as F
# from torch.utils.tensorboard import SummaryWriter

from sklearn import metrics
from sklearn.model_selection import RepeatedStratifiedKFold

from Utils.utils import *
from Utils.HGDataset import HGDataset

from Models.HAN import HAN
from Models.HGT import HGT
from torch_geometric.utils import from_scipy_sparse_matrix
from Utils.EarlyStopping import EarlyStopping

os.chdir("/private/xiongshuwen/CancerGene/workingspace/MyDriverGenes")


# os.environ['CUDA_VISIBLE_DEVICES'] = '1'


class Trainer:
    def __init__(self, model_name, epochs):
        super(Trainer, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
        self.feat_dim = 48
        self.hidden_channels = 64
        self.out_channels = 1
        self.num_heads = 4
        self.num_layers = 3
        self.lr = 0.01
        self.weight_decay = 5e-4
        # self.model = model.to(device=self.device)
        self.model = None
        self.model_name = model_name
        # self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=0.001, weight_decay=5e-4)
        self.optimizer = None
        # self.criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([2.7]).to(self.device))
        self.criterion = None
        self.epochs = epochs
        self.repeats = 1
        self.folds = 5

    def train(self, data, train_mask, val_mask, fold):

        early_stopping = EarlyStopping(patience=50, verbose=True)

        max_pr = 0

        for epoch in range(self.epochs):
            self.model.train()
            self.optimizer.zero_grad()
            out = self.model(data.x_dict, data.edge_index_dict)
            # Attention: Use the mask to get the label index, then use the label index to get the genes used to train
            pred = torch.sigmoid(out[data['gene'].label_index[train_mask]]).cpu().detach().numpy()
            # pred = np.round(
            #     torch.sigmoid(out[data['gene'].label_index[train_mask]]).cpu().detach().numpy())
            # pred = np.round(out.cpu().detach().numpy())
            train_loss = self.criterion(out[data['gene'].label_index[train_mask]].squeeze(), data['gene'].y[train_mask])
            labels = data['gene'].y[train_mask].cpu()
            train_ACC, train_F1, train_AUROC, train_AUPR = self.measure(pred, labels)
            train_loss.backward()
            self.optimizer.step()

            self.model.eval()
            with torch.no_grad():
                out = self.model(data.x_dict, data.edge_index_dict)
                pred = torch.sigmoid(out[data['gene'].label_index[val_mask]]).cpu().detach().numpy()
                # pred = np.round(
                #     torch.sigmoid(out[data['gene'].label_index[val_mask]]).cpu().detach().numpy())
                # pred = np.round(out.cpu().detach().numpy())
                val_loss = self.criterion(out[data['gene'].label_index[val_mask]].squeeze(), data['gene'].y[val_mask])
                labels = data['gene'].y[val_mask].cpu()
                val_ACC, val_F1, val_AUROC, val_AUPR = self.measure(pred, labels)


            # Save the best model
            model_path = os.path.join("SavedModels", self.model_name)
            make_dirs(model_path)
            stop = early_stopping(val_loss, self.model, os.path.join(model_path, "fold_{}.pth".format(fold)))
            if stop:
                break

            # print(
            #     "Round:{}, Fold:{}, Epoch:{}, Train_ACC:{}, Train_loss:{}, ValACC:{}, Val_F1:{}, Val_AUROC:{}, Val_AUPR:{}, Val_Loss:{}".format(
            #         (fold // 5) + 1, (fold % 5) + 1, epoch, np.round(train_ACC, 4), np.round(train_loss.item(), 4),
            #         np.round(val_ACC, 4), np.round(val_F1, 4),
            #         np.round(val_AUROC, 4), np.round(val_AUPR, 4), np.round(val_loss.item(), 4)))

        return val_ACC, val_F1, val_AUROC, val_AUPR, val_loss.item()

    def inference(self, data, mask, fold):
        path = os.path.join("./SavedModels", self.model_name, "fold_{}.pth".format(fold))
        self.model = torch.load(f=path, map_location='cpu')
        self.model = self.model.to(self.device)
        self.model.eval()
        out = self.model(data.x_dict, data.edge_index_dict)
        pred = torch.sigmoid(out[data['gene'].label_index[mask]]).cpu().detach().numpy()
        # pred = np.round(
        #     torch.sigmoid(out[data['gene'].label_index[mask]]).cpu().detach().numpy())
        # pred = np.round(out.cpu().detach().numpy())
        loss = self.criterion(out[data['gene'].label_index[mask]].squeeze(), data['gene'].y[mask])
        labels = data['gene'].y[mask].cpu()
        ACC, F1, AUROC, AUPR = self.measure(pred, labels)
        return pred, ACC, F1, AUROC, AUPR, loss.item()

    def measure(self, pred, labels):
        ACC = metrics.accuracy_score(y_true=labels, y_pred=np.round(pred))
        F1 = metrics.f1_score(y_true=labels, y_pred=np.round(pred))
        AUROC = metrics.roc_auc_score(y_true=labels, y_score=pred)
        precision, recall, _ = metrics.precision_recall_curve(labels, pred)
        AUPR = metrics.auc(recall, precision)
        return ACC, F1, AUROC, AUPR

    def save_evaluation_indicators(self, indicators):
        path = os.path.join("SavedIndicators")

        if not os.path.exists(path):
            os.makedirs(path)
        #     写入评价指标
        file_name = os.path.join(path, "{}.xlsx".format(self.model_name))
        file = open(file_name, "a")

        file.write(str(np.round(indicators[0], 4)) + " " + str(np.round(indicators[1], 4)) + " " +
                   str(np.round(indicators[2], 4)) + " " + str(np.round(indicators[3], 4)) + "\n")

        file.close()

    def run(self, data):

        kf = RepeatedStratifiedKFold(n_splits=self.folds, n_repeats=self.repeats, random_state=42)
        splits = kf.split(data['gene'].label_index, data['gene'].y)

        # pbar = tqdm(enumerate(splits))
        for fold, (train_mask, val_mask) in enumerate(splits):
            if self.model_name == 'HGT_re':
                self.model = HGT(hidden_channels=self.hidden_channels, out_channels=self.out_channels,
                                 num_heads=self.num_heads, num_layers=self.num_layers, data=data, fold=fold)
            elif self.model_name == 'HGT_without_residual3':
                self.model = HGT(hidden_channels=self.hidden_channels, out_channels=self.out_channels,
                                 num_heads=self.num_heads, num_layers=self.num_layers, data=data, fold=fold)
            elif self.model_name == 'HGT_get_attention':
                self.model = HGT(hidden_channels=self.hidden_channels, out_channels=self.out_channels,
                                 num_heads=self.num_heads, num_layers=self.num_layers, data=data, fold=fold)
            elif self.model_name == 'HAN3':
                self.model = HAN(in_channels=-1, out_channels=self.out_channels, hidden_size=self.hidden_channels,
                                 heads=self.num_heads, metadata=data.metadata())

            data, self.model = data.to(self.device), self.model.to(self.device)
            # data = data.to(self.device)
            self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.lr,
                                              weight_decay=self.weight_decay)
            self.criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([2.7]).to(self.device))

            with torch.no_grad():  # Initialize lazy modules.
                out = self.model(data.x_dict, data.edge_index_dict)

            train_ACC, train_F1, train_AUROC, train_AUPR, train_loss = self.train(data=data, train_mask=train_mask,
                                                                                  val_mask=val_mask, fold=fold)
            test_pred, test_ACC, test_F1, test_AUROC, test_AUPR, test_loss = self.inference(data=data, mask=val_mask,
                                                                                            fold=fold)

            # Save the indicators
            indicators = [test_ACC, test_F1, test_AUROC, test_AUPR]
            self.save_evaluation_indicators(indicators)


if __name__ == '__main__':
    model_name = 'HGT_without_residual3'
    # model_name = 'HGT_get_attention'
    # model_name = 'HAN3'
    dataset = HGDataset(root="/private/xiongshuwen/CancerGene/workingspace/MyDriverGenes")
    data = dataset[0]

    trainer = Trainer(model_name=model_name, epochs=1000)
    trainer.run(data=data)
