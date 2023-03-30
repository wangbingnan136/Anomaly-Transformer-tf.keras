
import stlearn as st
import scanpy as sc
import pandas as pd
import time
import os
import torch
from pathlib import Path
import glob2
from PIL import Image
import numpy as np
from torchvision import transforms
import sys
path='/root/autodl-tmp/multimodal/PathomicFusion-master/'
sys.path.append('/root/autodl-tmp/multimodal/PathomicFusion-master')


from CellGraph.model import CPC_model
from stMVC.utilities import parameter_setting
from stMVC.image_processing import tiling
# from make_split import split

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







# crop data into 64 by 64 with 32 overlap 
def cropdata(data, num_channels=3, kernel_size = 64, stride = 32):
    if len(data.shape) == 3:
        data = data.unsqueeze(0)

    data = data.unfold(2, kernel_size, stride).unfold(3, kernel_size, stride)
    data = data.permute(0,2,3,1,4,5)
    data = data.contiguous().view(-1, num_channels, kernel_size, kernel_size)
    return data
class args(object):
    basePath ='/root/autodl-tmp/multimodal/PathomicFusion-master/data/DLPFC/'
    use_cuda=True
    sizeImage=32 #图片大小处理为32X32的
    sizeImage2=40 #不对图片做resize，直接使用原图，原图是40X40的，尽量保持super resolution


def Preprocessing(args):
    start = time.time()

    dirs = os.listdir(args.basePath)
    for d in dirs:
        args.inputPath = args.basePath + d
        args.tillingPath = Path(args.inputPath + '/tmp/')
        args.tillingPath.mkdir(parents=True, exist_ok=True)
        
        args.tillingPath2 = Path(args.inputPath + '/tmp224/')
        args.tillingPath2.mkdir(parents=True, exist_ok=True)
        
        
        args.outPath = Path(args.inputPath+'/results/')
        args.outPath.mkdir(parents=True, exist_ok=True)

        # load spatial transcriptomics and histological data
        adata = sc.read_visium(args.inputPath)
        adata.var_names_make_unique()

        adata1 = adata.copy()

        sc.pp.normalize_total(adata, inplace=True)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=3000)

        adata2 = adata1[:, adata.var['highly_variable']]
        df = pd.DataFrame(adata2.X.toarray(), index=adata2.obs_names.tolist())
        df.to_csv(args.inputPath + '/HVG.csv')
        print('Successfully preprocessed {} genes and {} cells.'.format(adata2.n_vars, adata2.n_obs))

        args.use_cuda = args.use_cuda and torch.cuda.is_available()

        adata = st.convert_scanpy(adata)
        # save physical location of spots into Spot_location.csv file
        data = {'imagerow': adata.obs['imagerow'].values.tolist(), 'imagecol': adata.obs['imagecol'].values.tolist()}
        df = pd.DataFrame(data, index=adata.obs_names.tolist())
        df.to_csv(args.inputPath + '/spatial/' + 'Spot_location.csv')


        # tilling histological data and train sinCLR model
        print('Tilling spot image')
        tiling(adata, args.tillingPath, target_size=args.sizeImage)
        tiling(adata, args.tillingPath2, target_size=args.sizeImage2)
        # extract Graph features
        imgs = glob2.glob(str(args.tillingPath) + "/*.jpeg")
        img_feats = {}
        for img in imgs:
            img = img.rpartition("/")[-1]
            img_fname = img[:-7]
            img = Image.open(os.path.join(args.tillingPath, img))
            transform = transforms.Compose([transforms.Resize(256),transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            img_t = transform(img)
            img_t = img_t.unsqueeze(0)
            img_in = cropdata(img_t)
            feature = encoder(img_in.to(device)).cpu().detach().numpy()[0]
            # feature = encoder(img.to(device)).cpu().detach().numpy()[0]
            img_feats[img_fname] = feature
        img_feats = pd.DataFrame(data=img_feats).T
        img_feats.to_csv(args.inputPath + '/graph_features.csv')

    duration = time.time() - start
    print('Finish training, total time is: ' + str(duration) + 's')
    
    
if __name__=="__main__":
    
    device = torch.device('cuda:{}'.format('0'))
    model = CPC_model(1024, 256)
    encoder = model.encoder.to(device)
    ckpt_dir = path+'/CellGraph/cpc_glioma.pt'
    ckpt = torch.load(ckpt_dir)
    encoder.load_state_dict(ckpt['encoder_state_dict'])
    Preprocessing(args)
    
    Predictor = pl.Trainer(accelerator="gpu")
    vit_model = torch.load('/root/autodl-tmp/multimodal/PathomicFusion-master/cv_models/best_models/vit_best')

    knngraph = T.KNNGraph(36,loop=True,force_undirected=True,flow='source_to_target',cosine=False,num_workers=4)


    images = []
    labels = []
    tabulars = []
    graphs = []

    for index,ff in enumerate(os.listdir('/root/autodl-tmp/multimodal/PathomicFusion-master/data/DLPFC/')):
        if '151673' in ff:
            continue
        # data_path = '/root/autodl-tmp/multimodal/PathomicFusion-master/data/DLPFC/151673/tmp/'
        data_path = f'/root/autodl-tmp/multimodal/PathomicFusion-master/data/DLPFC/{ff}/tmp224/'

        files = os.listdir(data_path)

        label = pd.read_csv(f'/root/autodl-tmp/multimodal/PathomicFusion-master/data/DLPFC/{ff}/{ff}_annotation.txt',sep='\t')
        if 'Cell' in label.columns:
            label = label.set_index('Cell')
        tmp = list(label.index)
        tmp=[item.split('-')[0] for item in tmp]
        label.index = tmp

        graph_label = []
        images2graph = []
        glabel = label.Cluster.to_dict()

        label = label.query('Cluster!=0')#去掉无效样本
        label = label.Cluster.to_dict()

        dt = pd.read_csv(f'/root/autodl-tmp/multimodal/PathomicFusion-master/data/DLPFC/{ff}/HVG.csv')
        tab_index = []
        for i,file in tqdm(enumerate(files)):
            graph_label.append(glabel.get(file.split('-')[0]))
            images2graph.append(np.array(Image.open(data_path+file)))

            if label.get(file.split('-')[0]) is None:
                continue
            else:
                tab_index.append(i)
                images.append(np.array(Image.open(data_path+file)))
                labels.append(label.get(file.split('-')[0]))

        # tabdata = TabularDataset(ssd.transform(dt.drop('Unnamed: 0',axis=1)),np.zeros(dt.shape[0]))
        # tab_dataloader =  DataLoader(tabdata,num_workers=4,batch_size=1024,shuffle=False,drop_last =False)
        imgs = np.stack(images2graph,0)

        imgdata = ImageDataset(imgs,np.zeros(imgs.shape[0]),transform=prediction_transform)
        img_dataloader =  DataLoader(imgdata,num_workers=4,batch_size=128,shuffle=False,drop_last =False)

        fake_label = Predictor.predict(vit_model,img_dataloader)
        fake_label = torch.cat(fake_label)            

        dt = dt.iloc[tab_index]
        tabulars.append(dt)

        node_features = pd.read_csv(f'/root/autodl-tmp/multimodal/PathomicFusion-master/data/DLPFC/{ff}/graph_features.csv').drop('Unnamed: 0',axis=1)
        position = pd.read_csv(f'/root/autodl-tmp/multimodal/PathomicFusion-master/data/DLPFC/{ff}/spatial/Spot_location.csv')
        graph = knngraph(Data(x = torch.from_numpy(node_features.values).to(torch.float32),y=torch.as_tensor(graph_label).long(),pos =torch.from_numpy(position[['imagerow', 'imagecol']].values).to(torch.float32),name=ff,teacher_predictions = fake_label))
        graphs.append(graph)






    all_images = np.stack(images,axis=0)
    print('image shape',all_images.shape)
    all_labels = np.hstack(labels)
    all_labels = pd.Series(all_labels)
    print('label shape',all_labels.shape)
    np.save('unlabeled_all_images',all_images)
    dicts = dict(zip(range(1,8),range(7)))
    all_labels = all_labels.map(dicts)
    np.save('unlabeled_all_labels',all_labels.values)
    all_tabulars = pd.concat(tabulars,axis=0,ignore_index=True)
    all_tabulars.to_csv('unlabeled_all_tabular.csv',index=False)
    print('tabular shape',all_tabulars.shape)

    display(dicts)



    images = []
    labels = []
    tabulars = []

    for index,ff in enumerate(os.listdir('/root/autodl-tmp/multimodal/PathomicFusion-master/data/DLPFC/')):
        if '151673' not in ff:
            continue
        # data_path = '/root/autodl-tmp/multimodal/PathomicFusion-master/data/DLPFC/151673/tmp/'
        data_path = f'/root/autodl-tmp/multimodal/PathomicFusion-master/data/DLPFC/{ff}/tmp224/'

        files = os.listdir(data_path)

        label = pd.read_csv(f'/root/autodl-tmp/multimodal/PathomicFusion-master/data/DLPFC/{ff}/{ff}_annotation.txt',sep='\t')
        if 'Cell' in label.columns:
            label = label.set_index('Cell')
        tmp = list(label.index)
        tmp=[item.split('-')[0] for item in tmp]
        label.index = tmp

        graph_label = []
        images2graph = []
        glabel = label.Cluster.to_dict()

        label = label.query('Cluster!=0')#去掉无效样本
        label = label.Cluster.to_dict()

        dt = pd.read_csv(f'/root/autodl-tmp/multimodal/PathomicFusion-master/data/DLPFC/{ff}/HVG.csv')
        tab_index = []
        for i,file in tqdm(enumerate(files)):
            graph_label.append(glabel.get(file.split('-')[0]))
            images2graph.append(np.array(Image.open(data_path+file)))

            if label.get(file.split('-')[0]) is None:
                continue
            else:
                tab_index.append(i)
                images.append(np.array(Image.open(data_path+file)))
                labels.append(label.get(file.split('-')[0]))

        # tabdata = TabularDataset(ssd.transform(dt.drop('Unnamed: 0',axis=1)),np.zeros(dt.shape[0]))
        # tab_dataloader =  DataLoader(tabdata,num_workers=4,batch_size=1024,shuffle=False,drop_last =False)
        imgs = np.stack(images2graph,0)

        imgdata = ImageDataset(imgs,np.zeros(imgs.shape[0]),transform=prediction_transform)
        img_dataloader =  DataLoader(imgdata,num_workers=4,batch_size=128,shuffle=False,drop_last =False)

        fake_label = Predictor.predict(vit_model,img_dataloader)
        fake_label = torch.cat(fake_label)            

        dt = dt.iloc[tab_index]
        tabulars.append(dt)

        node_features = pd.read_csv(f'/root/autodl-tmp/multimodal/PathomicFusion-master/data/DLPFC/{ff}/graph_features.csv').drop('Unnamed: 0',axis=1)
        position = pd.read_csv(f'/root/autodl-tmp/multimodal/PathomicFusion-master/data/DLPFC/{ff}/spatial/Spot_location.csv')
        graph = knngraph(Data(x = torch.from_numpy(node_features.values).to(torch.float32),y=torch.as_tensor(graph_label).long(),pos =torch.from_numpy(position[['imagerow', 'imagecol']].values).to(torch.float32),name=ff,teacher_predictions = fake_label))
        graphs.append(graph)

    all_images = np.stack(images,axis=0)
    print('image shape',all_images.shape)
    all_labels = np.hstack(labels)
    all_labels = pd.Series(all_labels)
    print('label shape',all_labels.shape)
    np.save('labeled_all_images',all_images)
    dicts = dict(zip(range(1,8),range(7)))
    all_labels = all_labels.map(dicts)
    np.save('labeled_all_labels',all_labels.values)
    all_tabulars = pd.concat(tabulars,axis=0,ignore_index=True)
    all_tabulars.to_csv('labeled_all_tabular.csv',index=False)
    print('tabular shape',all_tabulars.shape)

    display(dicts)
    torch.save(graphs,'graphs')



    print(graphs)
    names = [graph.name for graph in graphs]
    print(names)