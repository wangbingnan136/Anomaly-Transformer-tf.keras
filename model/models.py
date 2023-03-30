import pandas as pd
import matplotlib.pyplot as plt
import rtdl

path = '/root/autodl-tmp/pytorch_lightning_text_classification/'
import torchmetrics
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import pytorch_lightning as pl
from pytorch_lightning.callbacks import *
import pytorch_lightning as pl
from torch.nn import functional as F
from torch import optim
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchlayers as tl
from torchmetrics.classification import Accuracy,Recall,Precision,F1Score
from pytorch_lightning.callbacks import ModelCheckpoint,LearningRateMonitor,RichProgressBar,EarlyStopping
import os
from livelossplot import PlotLosses
import pytorch_lightning as pl

from pytorch_lightning.loggers import CSVLogger

class LiveLossPlotCallback(pl.Callback):
    def __init__(self):
        super().__init__()
        self.liveplot = PlotLosses()

    def on_train_epoch_end(self, trainer, outputs):
        logs = {'log ' + k: v.item() for k, v in trainer.callback_metrics.items()}
        self.liveplot.update(logs)
        self.liveplot.send()
        return logs
    
    

from PIL import Image
import numpy as np
from tqdm import tqdm
import os
from torch.utils.data import Dataset,DataLoader
import torchvision.transforms as transforms
from sklearn.model_selection import StratifiedKFold,train_test_split
import torch
import pytorch_lightning as pl
from transformers import ViTModel, ViTConfig
from torch.optim import AdamW
import torchmetrics
from torch import nn
from torchvision import transforms
from pytorch_lightning.callbacks import RichProgressBar
# ,transforms.Resize((224, 224)),
import torch.nn.functional as F
torch.sparse_csc = torch.sparse_csr_tensor
import torch
import numpy as np
from torch.optim import Adam
from sklearn.model_selection import train_test_split
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import pytorch_lightning as pl

from pytorch_tabnet.tab_network import TabNet

from torch_geometric.data import Data,Batch
import torch_geometric.transforms as T




    
class ViTImageClassifier(pl.LightningModule):
    def __init__(self, num_classes: int, lr: float = 1e-4):
        super().__init__()
        
        metrics = torchmetrics.MetricCollection([
            # Accuracy: due to mode multiclass, not multilabel, this uses same formula as Precision
            torchmetrics.Accuracy(num_classes=num_classes,task='multiclass',average='micro'),
        ])

        self.train_metrics = metrics.clone(prefix='train_')
        self.valid_metrics = metrics.clone(prefix='val_')
        
        # 加载预训练的ViT模型和配置
        self.config = ViTConfig.from_pretrained('google/vit-base-patch16-224')
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224', config=self.config)

        # 添加分类器
        self.classifier =nn.Sequential(nn.Dropout(0.5),nn.Linear(self.config.hidden_size, num_classes))

        # 定义损失函数和优化器
        self.criterion = torch.nn.CrossEntropyLoss()
        # self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.lr = lr

    def forward(self, x):
        # 将输入图像传递给ViT模型
        outputs = self.vit(x)

        # 获取CLS token的输出，传递给分类器
        cls_token = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_token)

        return logits

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.flatten()
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        self.log('train_loss', loss)
        outputs = logits.softmax(-1)
        metrics = self.train_metrics(outputs, y)
        
        return loss
    
    def training_epoch_end(self, outputs):
        self.log_dict(self.train_metrics.compute(), on_step=False, on_epoch=True)
        self.train_metrics.reset()
        

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y.flatten()
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        self.log('val_loss', loss, prog_bar=True)
        outputs = logits.softmax(-1)
        metrics = self.valid_metrics(outputs, y) ##难道是因为这个。。。
        

    def validation_epoch_end(self, outputs):
        self.log_dict(self.valid_metrics.compute(), on_step=False, on_epoch=True)
        self.valid_metrics.reset()   
        
        
    def predict_step(self, batch, batch_idx):
        if isinstance(batch,(list,tuple)):
            x, y = batch
        else:
            x = batch
            
        logits = self.forward(x)
        return logits.softmax(-1)

    def configure_optimizers(self):
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.vit.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.01,'lr': self.lr},
            {'params': [p for n, p in self.vit.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0,'lr': 0.001},
        ]
        optimizer = AdamW(optimizer_grouped_parameters,lr=self.lr)

        return optimizer

    
class TabnetModel(pl.LightningModule):
    def __init__(self, num_features=3000, hidden_dim=256,num_classes=7,lr= 0.0005,tabnet=None):
        super().__init__()
        
        metrics = torchmetrics.MetricCollection([
            # Accuracy: due to mode multiclass, not multilabel, this uses same formula as Precision
            torchmetrics.Accuracy(num_classes=num_classes,task='multiclass',average='micro'),
        ])

        self.train_metrics = metrics.clone(prefix='train_')
        self.valid_metrics = metrics.clone(prefix='val_')

        
        self.tabnet = rtdl.ResNet.make_baseline(
    d_in=num_features,
    d_main=128,
    d_hidden=256,
    dropout_first=0.5,
    dropout_second=0.5,
    n_blocks=3,
    d_out=7,
)

        
        # 添加分类器
        # self.classifier =nn.Linear(hidden_dim, num_classes)

        # 定义损失函数和优化器
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
        # self.train_criterion = torch.nn.NLLLoss()
        # self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.lr = lr
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")

    def forward(self, x):
        # features,m_loss = self.tabnet(x)
        return self.tabnet(x)

#         logits = self.classifier(features)

#         return logits,m_loss


    def training_step(self, batch, batch_idx):
        x, y,fake = batch
        y = y.flatten()
        # logits,m_loss = self.forward(x)
        logits= self.forward(x)
        loss = self.criterion(logits.softmax(-1), y)
        index = torch.where(y==-1)[0]
        
        semi_logits = F.log_softmax(logits[index],dim=-1) ## torch的kl loss的坑爹设计，必须用log softmax
        teacher_logits = fake[index]
        semi_loss = self.kl_loss(semi_logits,teacher_logits)
        index = torch.where(y!=-1)[0]
        outputs = logits[index].softmax(-1)
        
        self.log('train_loss', loss)
        self.log('train_semi_loss', semi_loss)
        
        metrics = self.train_metrics(outputs, y[index])
        
        return loss+semi_loss
    
    
    def training_epoch_end(self, outputs):
        self.log_dict(self.train_metrics.compute(), on_step=False, on_epoch=True)
        self.train_metrics.reset()
        

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y.flatten()
        # logits,m_loss = self.forward(x)
        logits= self.forward(x)

        loss = self.criterion(logits.softmax(-1), y)#+0.001*m_loss
        self.log('val_loss', loss, prog_bar=True)
        outputs = logits.softmax(-1)
        metrics = self.valid_metrics(outputs, y) ##难道是因为这个。。。
        

    def validation_epoch_end(self, outputs):
        self.log_dict(self.valid_metrics.compute(), on_step=False, on_epoch=True)
        self.valid_metrics.reset()   
        
        
    def predict_step(self, batch, batch_idx):
        if isinstance(batch,(list,tuple)):
            x, y = batch
        else:
            x = batch
            
        logits = self.forward(x)
        return logits.softmax(-1)

    def configure_optimizers(self):

        optimizer = Adam(self.parameters(),lr=self.lr)

        return optimizer
    
    

from pytorch_tabnet.tab_network import TabNet
class TabNetBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self._build_network()

    def _build_network(self):
        # self.tabnet = TabNet(
        #     input_dim=self.hparams.continuous_dim + self.hparams.categorical_dim,
        #     output_dim=self.hparams.output_dim,
        #     n_d=self.hparams.n_d,
        #     n_a=self.hparams.n_a,
        #     n_steps=self.hparams.n_steps,
        #     gamma=self.hparams.gamma,
        #     cat_idxs=[i for i in range(self.hparams.categorical_dim)],
        #     cat_dims=[cardinality for cardinality, _ in self.hparams.embedding_dims],
        #     cat_emb_dim=[embed_dim for _, embed_dim in self.hparams.embedding_dims],
        #     n_independent=self.hparams.n_independent,
        #     n_shared=self.hparams.n_shared,
        #     epsilon=1e-15,
        #     virtual_batch_size=self.hparams.virtual_batch_size,
        #     momentum=0.02,
        #     mask_type=self.hparams.mask_type,
        # )
        self.tabnet =  TabNet(input_dim=3000,output_dim=256,n_d=16,
                       n_a=16,
                       n_steps=8,
                       gamma=1.08,
                       n_independent=6,virtual_batch_size=1024,
                       n_shared=2)

    # def unpack_input(self, x: Dict):
    #     # unpacking into a tuple
    #     x = x["categorical"], x["continuous"]
    #     # eliminating None in case there is no categorical or continuous columns
    #     x = (item for item in x if len(item) > 0)
    #     x = torch.cat(tuple(x), dim=1)
    #     return x

    def forward(self, x):
        # unpacking into a tuple
        # x = self.unpack_input(x)
        # Returns output and Masked Loss. We only need the output
        x, m_loss = self.tabnet(x)
        return x,m_loss
    
from torch_geometric.data.lightning import LightningNodeData
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.nn import BatchNorm1d
from torch_geometric.nn import GAT

import torch_geometric.data as geom_data
from torchmetrics import Accuracy 

class GATModel(pl.LightningModule):
    def __init__(self, in_channels: int, out_channels: int,
                 hidden_channels: int = 256, num_layers: int = 2,
                 dropout: float = 0.1,num_classes :int =7):
        super().__init__()
        self.gnn = GAT(in_channels, hidden_channels, num_layers,
                             out_channels, dropout=dropout,
                             norm=None).to(self.device)

        self.train_acc = Accuracy(num_classes=num_classes,task='multiclass',average='micro')
        self.val_acc = Accuracy(num_classes=num_classes,task='multiclass',average='micro')
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")


    def forward(self, x, edge_index):
        return self.gnn(x, edge_index)

    def training_step(self, data, batch_idx):
        y_hat = self.forward(data.x, data.edge_index)[:data.batch_size] ## 只考虑target node，其它node 不能放进来
        y = data.y[:data.batch_size].flatten()
        
        loss = self.criterion(y_hat, y)
        
        # idx = torch.where(y==-1)[0]
        # if idx.shape[0]==0:
        #     semi_loss = 0.0
        # else:
        
        idx = torch.arange(y.shape[0])
        semi_logits = F.log_softmax(y_hat[idx],dim=-1)
        semi_label = data.teacher_predictions[:data.batch_size][idx]
        semi_loss = self.kl_loss(semi_logits,semi_label)
            
        self.log('train_loss',loss, prog_bar=True, on_step=False,
                 on_epoch=True)        
        self.log('train_semi_loss',semi_loss, prog_bar=True, on_step=False,
                 on_epoch=True)     
        
        idx = torch.where(y!=-1)[0]
        if idx.shape[0]==0:
            pass
        else:
        
            self.train_acc(y_hat.softmax(dim=-1)[idx], y[idx])
            self.log('train_acc', self.train_acc, prog_bar=True, on_step=False,
                     on_epoch=True,batch_size=idx.shape[0])
        return loss+semi_loss

    def validation_step(self, data, batch_idx):
        y_hat = self(data.x, data.edge_index)[:data.batch_size]
        y = data.y[:data.batch_size].flatten()
        idx = torch.where(y!=-1)[0]
        self.val_acc(y_hat.softmax(dim=-1)[idx], y[idx])
        self.log('val_acc', self.val_acc, prog_bar=True, on_step=False,
                 on_epoch=True,batch_size=idx.shape[0])
        
    def predict_step(self,data,batch_idx):
        y_hat = self(data.x, data.edge_index)[:data.batch_size]
        return y_hat.softmax(dim=-1)
        


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)
    
    
    
# datamodule = LightningNodeData(
#     big_graph,
#     input_train_nodes=big_graph.train_mask,
#     input_val_nodes=big_graph.val_mask,
#     loader='neighbor',
#     num_neighbors=[-1,-1],
#     batch_size=1024,
#     num_workers=8,
# )

import math
def init_max_weights(module):
    for m in module.modules():
        if type(m) == nn.Linear:
            stdv = 1. / math.sqrt(m.weight.size(1))
            m.weight.data.normal_(0, stdv)
            m.bias.data.zero_()
            
class TrilinearFusion_B(nn.Module):
    def __init__(self, skip=1, use_bilinear=1, gate1=1, gate2=1, gate3=1, dim1=7, dim2=7, dim3=7, scale_dim1=1, scale_dim2=1, scale_dim3=1, mmhid=96, dropout_rate=0.25,device='cpu'):
        super(TrilinearFusion_B, self).__init__()
        self.skip = skip
        self.use_bilinear = use_bilinear
        self.gate1 = gate1
        self.gate2 = gate2
        self.gate3 = gate3

        dim1_og, dim2_og, dim3_og, dim1, dim2, dim3 = dim1, dim2, dim3, dim1//scale_dim1, dim2//scale_dim2, dim3//scale_dim3
        skip_dim = dim1+dim2+dim3+3 if skip else 0

        # Path
        self.linear_h1 = nn.Sequential(nn.Linear(dim1_og, dim1), nn.ReLU())
        self.linear_z1 = nn.Bilinear(dim1_og, dim3_og, dim1) if use_bilinear else nn.Sequential(nn.Linear(dim1_og+dim3_og, dim1))
        self.linear_o1 = nn.Sequential(nn.Linear(dim1, dim1), nn.ReLU(), nn.Dropout(p=dropout_rate))

        # Graph
        self.linear_h2 = nn.Sequential(nn.Linear(dim2_og, dim2), nn.ReLU())
        self.linear_z2 = nn.Bilinear(dim2_og, dim1_og, dim2) if use_bilinear else nn.Sequential(nn.Linear(dim2_og+dim1_og, dim2))
        self.linear_o2 = nn.Sequential(nn.Linear(dim2, dim2), nn.ReLU(), nn.Dropout(p=dropout_rate))

        # Omic
        self.linear_h3 = nn.Sequential(nn.Linear(dim3_og, dim3), nn.ReLU())
        self.linear_z3 = nn.Bilinear(dim1_og, dim3_og, dim3) if use_bilinear else nn.Sequential(nn.Linear(dim1_og+dim3_og, dim3))
        self.linear_o3 = nn.Sequential(nn.Linear(dim3, dim3), nn.ReLU(), nn.Dropout(p=dropout_rate))

        self.post_fusion_dropout = nn.Dropout(p=0.25)
        self.encoder1 = nn.Sequential(nn.Linear(512, mmhid), nn.ReLU(), nn.Dropout(p=dropout_rate))
        self.encoder2 = nn.Sequential(nn.Linear(120, mmhid), nn.ReLU(), nn.Dropout(p=dropout_rate))
        self.classifier = nn.Linear(mmhid,7)
        self.weights = nn.Parameter(torch.ones(4))
        
        init_max_weights(self)
        self.device = device

    def forward(self, vec,return_fusion_embedding=True,return_gate_weights=True):
        vec1, vec2, vec3 = vec
        # vec1 = torch.log(vec1)
        # vec2 = torch.log(vec2)
        # vec3 = torch.log(vec3)
        
        #vec[:,0,:],vec[:,1,:],vec[:,2,:]
        # Gated Multimodal Units
        if self.gate1:
            h1 = self.linear_h1(vec1)
            z1 = self.linear_z1(vec1, vec3) if self.use_bilinear else self.linear_z1(torch.cat((vec1, vec3), dim=1))
            # Gate Path with Omic
            o1 = self.linear_o1(nn.Sigmoid()(z1)*h1)
        else:
            o1 = self.linear_o1(vec1)

        if self.gate2:
            h2 = self.linear_h2(vec2)
            z2 = self.linear_z2(vec2, vec1) if self.use_bilinear else self.linear_z2(torch.cat((vec2, vec1), dim=1))
            # Gate Graph with Omic
            o2 = self.linear_o2(nn.Sigmoid()(z2)*h2)
        else:
            o2 = self.linear_o2(vec2)

        if self.gate3:
            h3 = self.linear_h3(vec3)
            z3 = self.linear_z3(vec1, vec3) if self.use_bilinear else self.linear_z3(torch.cat((vec1, vec3), dim=1))
            # Gate Omic With Path
            o3 = self.linear_o3(nn.Sigmoid()(z3)*h3)
        else:
            o3 = self.linear_o3(vec3)

        # # Fusion
        # o1 = o1.view(-1,1)
        # o2 = o2.view(-1,1)
        # o3 = o3.view(-1,1)
        if self.device!='cpu':
        
            o1 = torch.cat((o1, torch.cuda.FloatTensor(o1.shape[0], 1).fill_(1)), 1)
            o2 = torch.cat((o2, torch.cuda.FloatTensor(o2.shape[0], 1).fill_(1)), 1)
            o3 = torch.cat((o3, torch.cuda.FloatTensor(o3.shape[0], 1).fill_(1)), 1)
        else:
            o1 = torch.cat((o1, torch.FloatTensor(o1.shape[0], 1).fill_(1)), 1)
            o2 = torch.cat((o2, torch.FloatTensor(o2.shape[0], 1).fill_(1)), 1)
            o3 = torch.cat((o3, torch.FloatTensor(o3.shape[0], 1).fill_(1)), 1)
        
        
        o12 = torch.bmm(o1.unsqueeze(2), o2.unsqueeze(1)).flatten(start_dim=1)
        o123 = torch.bmm(o12.unsqueeze(2), o3.unsqueeze(1)).flatten(start_dim=1)
        out = self.post_fusion_dropout(o123)
        out = self.encoder1(out)
        if self.skip: 
            out = torch.cat((out, o1, o2, o3), 1)
        out = self.encoder2(out)
        # stacks = torch.stack([self.classifier(out).softmax(-1),vec1,vec2,vec3],dim=-1)
        # alpha = torch.softmax(self.weights, dim=-1)
        # out_weighted = torch.einsum('abc,c->ab',stacks,alpha)
        # return out_weighted
        weights = torch.relu(self.weights)
        vec4 = F.log_softmax(self.classifier(out),-1)
        results = self.weights[0]*vec1+self.weights[1]*vec2+self.weights[2]*vec3+self.weights[3]*vec4
        final_return = [results]
        if return_fusion_embedding:
            final_return.append(vec4)
        if return_gate_weights:
            final_return.append([z1,z2,z3])
        return final_return
        