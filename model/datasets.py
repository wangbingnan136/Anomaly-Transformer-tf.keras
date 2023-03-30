from torch.utils.data import Dataset
import torch
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



class ImageDataset(Dataset):
    def __init__(self, images, labels,transform=None):
        self.images = images
        self.labels = labels.reshape(-1,1)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        if self.transform:
            img = self.transform(image=img)['image']

        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return img, label
    
    
class TabularDataset(Dataset):
    def __init__(self, datas, labels,teacher_predictions=None):
        self.datas = datas
        self.labels = labels.reshape(-1,1)
        self.fake_logits = teacher_predictions

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        data = torch.from_numpy(self.datas[idx]).to(torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        if self.fake_logits is not None:
            return data,label,torch.from_numpy(self.fake_logits[idx]).to(torch.float32)
        else:
            return data, label
        
class MultiModalDataset(Dataset):
    def __init__(self, image_embeddings,tabular_embeddings,graph_embeddings, labels):
        self.images = image_embeddings
        self.tabular = tabular_embeddings
        self.graph = graph_embeddings
        self.labels = labels.reshape(-1,1)
        

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        tabular = self.tabular[idx]
        graph = self.graph[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return img,tabular,graph, label