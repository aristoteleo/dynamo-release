import os
import sys
import glob
import shutil
import datetime
import pandas as pd
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import *
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.neighbors import NearestNeighbors
from joblib import Parallel, delayed
from tqdm import tqdm
import pkg_resources
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.simplefilter("ignore", UserWarning)
import logging
handle='cellDancer'
logger_cd=logging.getLogger(handle)
logging.getLogger(handle).setLevel(logging.INFO)

logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
from .sampling import *

class DNN_layer(nn.Module):

    """Define network structure.
    """

    def __init__(self, h1, h2):
        super().__init__()
        self.l1 = nn.Linear(2, h1)
        self.l2 = nn.Linear(h1, h2)
        self.l3 = nn.Linear(h2, 3)

    def forward(self, unsplice, splice, alpha0, beta0, gamma0, dt):
        #print(f"dt is {dt}")
        input = torch.tensor(np.array([np.array(unsplice), np.array(splice)]).T)
        x = self.l1(input)
        x = F.leaky_relu(x)
        x = self.l2(x)
        x = F.leaky_relu(x)
        x = self.l3(x)
        output = torch.sigmoid(x)
        beta = output[:,0]
        gamma = output[:,1]
        alphas = output[:,2]

        alphas = alphas * alpha0
        beta =  beta * beta0
        gamma = gamma * gamma0

        unsplice_predict = unsplice + (alphas - beta*unsplice)*dt
        splice_predict = splice + (beta*unsplice - gamma*splice)*dt
        return unsplice_predict, splice_predict, alphas, beta, gamma

    def save(self, model_path):
        torch.save({
            "l1": self.l1,
            "l2": self.l2,
            "l3": self.l3
        }, model_path)

    def load(self, model_path):
        checkpoint = torch.load(model_path)
        self.l1 = checkpoint["l1"]
        self.l2 = checkpoint["l2"]
        self.l3 = checkpoint["l3"]

class DNN_module(nn.Module):
    '''
    calculate loss function
    load network "DNN_layer"
    predict splice_predict and unsplice_predict
    '''
    def __init__(self, module, n_neighbors = None):
        super().__init__()
        self.module = module
        self.n_neighbors = n_neighbors

    def velocity_calculate(self, 
                           unsplice, 
                           splice, 
                           alpha0, 
                           beta0, 
                           gamma0,
                           dt,
                           embedding1,
                           embedding2, 
                           barcode = None, 
                           loss_func = None,
                           cost2_cutoff=None,
                           trace_cost_ratio=None,
                           corrcoef_cost_ratio=None):
        '''
        add embedding
        for real dataset
        calculate loss function
        predict unsplice_predict splice_predict from network 
        '''
        #generate neighbor indices and expr dataframe
        points = np.array([embedding1.numpy(), embedding2.numpy()]).transpose()

        self.n_neighbors=min((points.shape[0]-1), self.n_neighbors)
        nbrs = NearestNeighbors(n_neighbors=self.n_neighbors, algorithm='ball_tree').fit(points)
        
        distances, indices = nbrs.kneighbors(points) 
        # indices: 
        #   row -> cell, 
        #   col -> neighboring cells, 
        #   value -> index of cells, 
        #   the fist col is the index of row

        expr = pd.merge(pd.DataFrame(splice, columns=['splice']), pd.DataFrame(unsplice, columns=['unsplice']), left_index=True, right_index=True)
        if barcode is not None:
            expr.index = barcode
        unsplice = torch.tensor(expr['unsplice'])
        splice = torch.tensor(expr['splice'])
        indices = torch.tensor(indices)
        unsplice_predict, splice_predict, alphas, beta, gamma = self.module(unsplice, splice, alpha0, beta0, gamma0, dt)

        def cosine_similarity(unsplice, splice, unsplice_predict, splice_predict, indices):
            """Cost function
            Return:
                list of cosine distance and a list of the index of the next cell
            """
            
            uv, sv = unsplice_predict-unsplice, splice_predict-splice # Velocity from (unsplice, splice) to (unsplice_predict, splice_predict)
            unv, snv = unsplice[indices.T[1:]] - unsplice, splice[indices.T[1:]] - splice # Velocity from (unsplice, splice) to its neighbors

            den = torch.sqrt(unv**2 + snv**2) * torch.sqrt(uv**2+sv**2)
            den[den==0] = -1
            cosine = torch.where(den!=-1, (unv*uv + snv*sv) / den, torch.tensor(1.)) # cosine: column -> individuel cell (cellI); row -> nearby cells of cell id ; value -> cosine between col and row cells
            cosine_max, cosine_max_idx = torch.max(cosine, dim=0)
            cell_idx = torch.diag(indices[:, cosine_max_idx+1])
            return 1 - cosine_max, cell_idx



        def rmse(unsplice, splice, unsplice_predict, splice_predict, indices):
            """
            This loss is defined as the rmse of the predicted velocity vector (uv, sv) from the neighboring velocity vectors (unv, snv).

            This loss is used during revision.

            """
            uv, sv = unsplice_predict-unsplice, splice_predict-splice 
            unv, snv = unsplice[indices.T[1:]] - unsplice, splice[indices.T[1:]] - splice 

            rmse = (uv-unv)**2 + (sv-snv)**2
            rmse = torch.sqrt(0.5*rmse)

            # normalize across all neighboring cells using a softmax function.
            # m = torch.nn.Softmax(dim=0)
            # rmse = m(rmse)

            rmse_min, rmse_min_idx = torch.min(rmse, dim=0)
            cell_idx = torch.diag(indices[:, rmse_min_idx+1])
            return rmse_min, cell_idx


        def mix_loss(unsplice, splice, unsplice_predict, splice_predict, indices, mix_ratio = 0.5):
            """
            This loss is defined as the mix of rmse loss and cosine loss.

            This loss is used during revision.

            Parameters:
            
            unsplice: 1d tensor [n_cells] 
            splice: 1d tensor [n_cells] 
            indices: 2d array [n_cells, n_neighbors]
            Return:
                list of cosine distance and a list of the index of the next cell
            """

            #print("mix ratio, ", mix_ratio)
            uv, sv = unsplice_predict-unsplice, splice_predict-splice 
            unv, snv = unsplice[indices.T[1:]] - unsplice, splice[indices.T[1:]] - splice 
            mag_v = torch.sqrt(uv**2 + sv**2)
            mag_nv = torch.sqrt(unv**2 + snv**2)
            mag = (mag_nv - mag_v)**2

            # minimize mag or maximize -mag
            # normalize across all neighboring cells using a softmax function
            m = torch.nn.Softmax(dim=0)
            mag = m(mag)

            den = mag_v * mag_nv
            den[den==0] = -1

            # cosine: [n_neighbors x n_cells]
            cosine = torch.where(den!=-1, (unv*uv + snv*sv) / den, torch.tensor(1.))

            total = mix_ratio*(1-cosine) + (1 - mix_ratio)* mag
            total_min, total_min_idx = torch.min(total, dim=0)

            cell_idx = torch.diag(indices[:, total_min_idx+1])
            return total_min, cell_idx

        
        def trace_cost(unsplice, splice, unsplice_predict, splice_predict, idx, version):

            # This cost has been deprecated.

            uv, sv = unsplice_predict-unsplice, splice_predict-splice
            tan = torch.where(sv!=1000000, uv/sv, torch.tensor(0.00001))
            atan_theta = torch.atan(tan) + torch.pi/2
            atan_theta2=atan_theta[idx]
            atan_theta3 = atan_theta[idx[idx]]
            if version=="v1":
                cost = atan_theta2/atan_theta+atan_theta3/atan_theta2
            elif version=="v2":
                cost=torch.where(atan_theta<atan_theta2, 1, 0)+torch.where(atan_theta2<atan_theta3, 1, 0) 
                
            return(cost)

        def corrcoef_cost(alphas, unsplice, beta, splice):

            # This cost has been deprecated.
            
            corrcoef1 = torch.corrcoef(torch.tensor([alphas.detach().numpy(),unsplice.detach().numpy()]))[1,0]
            corrcoef2 = torch.corrcoef(torch.tensor([beta.detach().numpy(), splice.detach().numpy()]))[1,0]
            corrcoef = corrcoef1 + corrcoef2
            cost=torch.where(corrcoef>=torch.tensor(0.0), torch.tensor(0.0), torch.tensor(-corrcoef))
            return(cost)
        
        if trace_cost_ratio == 0 and corrcoef_cost_ratio == 0:

            if loss_func == 'cosine':
                cost1 = cosine_similarity(unsplice, splice, unsplice_predict, splice_predict, indices)[0]
                cost_fin = torch.mean(cost1)

            if loss_func == 'rmse':
                cost1 = rmse(unsplice, splice, unsplice_predict, splice_predict, indices)[0]
                cost_fin = torch.mean(cost1)

            elif 'mix' in loss_func:
                mix_ratio = loss_func[1]
                cost1 = mix_loss(unsplice, splice, unsplice_predict, splice_predict, indices, mix_ratio=mix_ratio)[0]
                cost_fin = torch.mean(cost1)

        else: # trace cost and corrcoef cost have been deprecated.
            # cosine cost
            cost1,idx = cosine_similarity(unsplice, splice, unsplice_predict, splice_predict, indices)
            cost1_normalize=(cost1-torch.min(cost1))/torch.max(cost1)
            cost1_mean = torch.mean(cost1_normalize)

            # trace cost
            if trace_cost_ratio>0:
                cost2 = trace_cost(unsplice, splice, unsplice_predict, splice_predict, idx,"v2")
                cost2_normalize=(cost2-torch.min(cost2))/torch.max(cost2)
                cost2_mean = torch.mean(cost2_normalize)
                cost2_relu=(max((cost2_mean-cost2_cutoff), 0))

            # corrcoef cost
            if corrcoef_cost_ratio>0:
                corrcoef_cost=corrcoef_cost(alphas, unsplice, beta, splice)

            # sum all cost
            cosin_cost_ratio=1-trace_cost_ratio-corrcoef_cost_ratio
            cost_fin = cosin_cost_ratio*cost1_mean + \
                       trace_cost_ratio*cost2_relu + \
                       corrcoef_cost_ratio*corrcoef_cost
            
        return cost_fin, unsplice_predict, splice_predict, alphas, beta, gamma


    def summary_para_validation(self, cost_mean): 
        loss_df = pd.DataFrame({'cost': cost_mean}, index=[0])
        return(loss_df)

    def summary_para(self, unsplice, splice, unsplice_predict, splice_predict, alphas, beta, gamma, cost): 
        cellDancer_df = pd.merge(pd.DataFrame(unsplice, columns=['unsplice']),pd.DataFrame(splice, columns=['splice']), left_index=True, right_index=True) 
        cellDancer_df['unsplice_predict'] = unsplice_predict
        cellDancer_df['splice_predict'] = splice_predict
        cellDancer_df['alpha'] = alphas
        cellDancer_df['beta'] = beta
        cellDancer_df['gamma'] = gamma
        cellDancer_df['cost'] = cost
        return cellDancer_df

class ltModule(pl.LightningModule):
    '''
    train network using "DNN_module"
    '''
    def __init__(self, 
                backbone=None, 
                initial_zoom=2, 
                initial_strech=1,
                learning_rate=None,
                dt=None,
                loss_func = None,
                cost2_cutoff=0,
                optimizer='Adam',
                trace_cost_ratio=0,
                corrcoef_cost_ratio=0,
                cost_type='smooth',
                average_cost_window_size=10,
                smooth_weight=0.9):
        super().__init__()
        self.backbone = backbone
        self.validation_loss_df = pd.DataFrame()
        self.test_cellDancer_df = None
        self.test_loss_df = None
        self.initial_zoom = initial_zoom
        self.initial_strech = initial_strech
        self.learning_rate=learning_rate
        self.dt=dt
        self.loss_func=loss_func
        self.cost2_cutoff=cost2_cutoff
        self.optimizer=optimizer
        self.trace_cost_ratio=trace_cost_ratio
        self.corrcoef_cost_ratio=corrcoef_cost_ratio
        self.save_hyperparameters()
        self.get_loss=1000
        self.cost_type=cost_type
        self.average_cost_window_size=average_cost_window_size # will be used only when cost_tpye.isin(['average', 'median'])
        self.cost_window=[]
        self.smooth_weight=smooth_weight
        
    def save(self, model_path):
        self.backbone.module.save(model_path)    # save network

    def load(self, model_path):
        self.backbone.module.load(model_path)   # load network

    def configure_optimizers(self):     # define optimizer
        if self.optimizer=="Adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, betas=(0.9, 0.999), eps=10**(-8), weight_decay=0.004, amsgrad=False)
        elif self.optimizer=="SGD":
            optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.8)
        return optimizer

    def training_step(self, batch, batch_idx):
        '''
        traning network
        batch: [] output returned from realDataset.__getitem__
        
        '''

        unsplices, splices, gene_names, unsplicemaxs, splicemaxs, embedding1s, embedding2s = batch
        unsplice, splice, unsplicemax, splicemax, embedding1, embedding2  = unsplices[0], splices[0], unsplicemaxs[0], splicemaxs[0], embedding1s[0], embedding2s[0]
        
        umax = unsplicemax
        smax = splicemax
        alpha0 = np.float32(umax*self.initial_zoom)
        beta0 = np.float32(1.0)
        gamma0 = np.float32(umax/smax*self.initial_strech)

        cost, unsplice_predict, splice_predict, alphas, beta, gamma = self.backbone.velocity_calculate( \
                unsplice, splice, alpha0, beta0, gamma0, self.dt, embedding1, embedding2, \
                loss_func = self.loss_func, \
                cost2_cutoff = self.cost2_cutoff, \
                trace_cost_ratio = self.trace_cost_ratio, \
                corrcoef_cost_ratio=self.corrcoef_cost_ratio)

        if self.cost_type=='average': # keep the window len <= check_val_every_n_epoch
            if len(self.cost_window)<self.average_cost_window_size:
                self.cost_window.append(cost)
            else:
                self.cost_window.pop(0)
                self.cost_window.append(cost)
            self.get_loss = torch.mean(torch.stack(self.cost_window))
            self.log("loss", self.get_loss)
            
        elif self.cost_type=='median': # keep the window len <= check_val_every_n_epoch
            if len(self.cost_window)<self.average_cost_window_size:
                self.cost_window.append(cost)
            else:
                self.cost_window.pop(0)
                self.cost_window.append(cost)
            self.get_loss = torch.median(torch.stack(self.cost_window))
            self.log("loss", self.get_loss)
            
        elif self.cost_type=='smooth':
            if self.get_loss==1000:
                self.get_loss=cost
            smoothed_val = cost * self.smooth_weight + (1 - self.smooth_weight) * self.get_loss  # calculate smoothed value
            self.get_loss = smoothed_val  
            self.log("loss", self.get_loss)
        else:
            self.get_loss = cost
            self.log("loss", self.get_loss) 
        
        return {
            "loss": cost,
            "beta": beta.detach(),
            "gamma": gamma.detach()
        }

    def training_epoch_end(self, outputs):
        '''
        steps after finished each epoch
        '''
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        beta = torch.stack([x["beta"] for x in outputs]).mean()
        gamma = torch.stack([x["gamma"] for x in outputs]).mean()

    def validation_step(self, batch, batch_idx):
        '''
        predict unsplice_predict, splice_predict on the training dataset
        '''

        unsplices, splices, gene_names, unsplicemaxs, splicemaxs, embedding1s, embedding2s = batch
        unsplice, splice,gene_name, unsplicemax, splicemax, embedding1, embedding2  = unsplices[0], splices[0], gene_names[0], unsplicemaxs[0], splicemaxs[0], embedding1s[0], embedding2s[0]
        if self.current_epoch!=0:
            cost = self.get_loss.data.numpy()
            loss_df = self.backbone.summary_para_validation(cost)
            loss_df.insert(0, "gene_name", gene_name)
            loss_df.insert(1, "epoch", self.current_epoch)
            if self.validation_loss_df.empty:
                self.validation_loss_df = loss_df
            else:
                self.validation_loss_df = self.validation_loss_df.append(loss_df)

    def test_step(self, batch, batch_idx):
        unsplices, splices, gene_names, unsplicemaxs, splicemaxs, embedding1s, embedding2s = batch
        unsplice, splice, gene_name, unsplicemax, splicemax, embedding1, embedding2  = unsplices[0], splices[0], gene_names[0], unsplicemaxs[0], splicemaxs[0], embedding1s[0], embedding2s[0]
        umax = unsplicemax
        smax = splicemax
        alpha0 = np.float32(umax*2)
        beta0 = np.float32(1.0)
        gamma0 = np.float32(umax/smax)

        cost, unsplice_predict, splice_predict, alphas, beta, gamma = self.backbone.velocity_calculate( \
                unsplice, splice, alpha0, beta0, gamma0, self.dt, embedding1, embedding2, \
                loss_func = self.loss_func, \
                cost2_cutoff = self.cost2_cutoff, \
                trace_cost_ratio = self.trace_cost_ratio, \
                corrcoef_cost_ratio=self.corrcoef_cost_ratio)

        self.test_cellDancer_df= self.backbone.summary_para(
            unsplice, splice, unsplice_predict.data.numpy(), splice_predict.data.numpy(), 
            alphas.data.numpy(), beta.data.numpy(), gamma.data.numpy(), 
            cost.data.numpy())
        
        self.test_cellDancer_df.insert(0, "gene_name", gene_name)
        self.test_cellDancer_df.insert(0, "cellIndex", self.test_cellDancer_df.index)


class getItem(Dataset): 
    def __init__(self, data_fit=None, data_predict=None,datastatus="predict_dataset", permutation_ratio=0.1,norm_u_s=True,norm_cell_distribution=False): 
        self.data_fit=data_fit
        self.data_predict=data_predict
        self.datastatus=datastatus
        self.permutation_ratio=permutation_ratio
        self.gene_name=list(data_fit.gene_name.drop_duplicates())
        self.norm_u_s=norm_u_s
        self.norm_max_unsplice=None
        self.norm_max_splice=None
        self.norm_cell_distribution=norm_cell_distribution

    def __len__(self):
        return len(self.gene_name) # gene count

    def __getitem__(self, idx):
        gene_name = self.gene_name[idx]

        if self.datastatus=="fit_dataset":
            data_fitting=self.data_fit[self.data_fit.gene_name==gene_name] # unsplice & splice for cells for one gene
            if self.norm_cell_distribution==True:    # select cells to train using norm_cell_distribution methods
                unsplice = data_fitting.unsplice
                splice = data_fitting.splice
                unsplicemax_fit = np.float32(max(unsplice))
                splicemax_fit = np.float32(max(splice))
                unsplice = np.round(unsplice/unsplicemax_fit, 2)*unsplicemax_fit
                splice = np.round(splice/splicemax_fit, 2)*splicemax_fit
                upoints = np.unique(np.array([unsplice, splice]), axis=1)
                unsplice = upoints[0]
                splice = upoints[1]
                data_fitting = pd.DataFrame({'gene_name':gene_name,'unsplice':unsplice, 'splice':splice,'embedding1':unsplice,'embedding2':splice})
        
            # random sampling in each epoch
            if self.permutation_ratio==1:
                data=data_fitting
            elif (self.permutation_ratio<1) & (self.permutation_ratio>0):
                data=data_fitting.sample(frac=self.permutation_ratio)  # select cells to train using random methods
            else:
                print('sampling ratio is wrong!')
        elif self.datastatus=="predict_dataset":
            data_pred=self.data_predict[self.data_predict.gene_name==gene_name] # unsplice & splice for cells for one gene
            data=data_pred
            
        data_pred=self.data_predict[self.data_predict.gene_name==gene_name] # unsplice & splice for cells for one gene

        unsplicemax = np.float32(max(data_pred["unsplice"]))
        splicemax = np.float32(max(data_pred["splice"]))
        unsplice = np.array(data.unsplice.copy().astype(np.float32))
        splice = np.array(data.splice.copy().astype(np.float32))
        if self.norm_u_s:
            unsplice=unsplice/unsplicemax
            splice=splice/splicemax

        # add embedding
        embedding1 = np.array(data.embedding1.copy().astype(np.float32))
        embedding2 = np.array(data.embedding2.copy().astype(np.float32))

        return unsplice, splice, gene_name, unsplicemax, splicemax, embedding1, embedding2



class feedData(pl.LightningDataModule):
    '''
    load training and test data
    '''
    def __init__(self, data_fit=None, data_predict=None,permutation_ratio=1,norm_u_s=True,norm_cell_distribution=False):
        super().__init__()

        self.fit_dataset = getItem(data_fit=data_fit, data_predict=data_predict,datastatus="fit_dataset", permutation_ratio=permutation_ratio,norm_u_s=norm_u_s,norm_cell_distribution=norm_cell_distribution)
        
        self.predict_dataset = getItem(data_fit=data_fit, data_predict=data_predict,datastatus="predict_dataset", permutation_ratio=permutation_ratio,norm_u_s=norm_u_s)

    def subset(self, indices):
        import copy
        temp = copy.copy(self)
        temp.fit_dataset = Subset(self.fit_dataset, indices)
        temp.predict_dataset = Subset(self.predict_dataset, indices)
        return temp

    def train_dataloader(self):
        return DataLoader(self.fit_dataset,num_workers=0)
    def val_dataloader(self):
        return DataLoader(self.fit_dataset,num_workers=0)
    def test_dataloader(self):
        return DataLoader(self.predict_dataset,num_workers=0,)

def _train_thread(datamodule, 
                  data_indices,
                  save_path=None,
                  max_epoches=None,
                  check_val_every_n_epoch=None,
                  norm_u_s=None,
                  patience=None,
                  learning_rate=None,
                  dt=None,
                  loss_func=None,
                  n_neighbors=None,
                  ini_model=None,
                  model_save_path=None):
    
    try:
        seed = 0
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        # iniate network (DNN_layer) and loss function (DynamicModule)
        backbone = DNN_module(DNN_layer(100, 100), n_neighbors=n_neighbors)
        model = ltModule(backbone=backbone, dt=dt, learning_rate=learning_rate, loss_func=loss_func)

        selected_data = datamodule.subset(data_indices)

        unsplice, splice, this_gene_name, unsplicemax, splicemax, embedding1, embedding2=selected_data.fit_dataset.__getitem__(0)

        data_df=pd.DataFrame({'unsplice':unsplice,'splice':splice,'embedding1':embedding1,'embedding2':embedding2})
        data_df['gene_name']=this_gene_name
        try:

            # Note
            # here n_neighbors in the downsampling_embedding function is for selecting initial model.
            # which is different from the n_neighbors in _train_tread for velocity calculation.
            _, sampling_ixs_select_model, _ = downsampling_embedding(data_df, # for select model
                                para='neighbors',
                                step=(20,20),
                                n_neighbors=30,
                                target_amount=None,
                                projection_neighbor_choice='embedding')
        except:
            sampling_ixs_select_model=list(data_df.index)
            
        gene_downsampling=downsampling(data_df=data_df, gene_list=[this_gene_name], downsampling_ixs=sampling_ixs_select_model)
        if ini_model=='circle':
            model_path=model_path=pkg_resources.resource_stream(__name__,os.path.join('model', 'circle.pt')).name
        if ini_model=='branch':
            model_path=model_path=pkg_resources.resource_stream(__name__,os.path.join('model', 'branch.pt')).name
        else:
            model_path=select_initial_net(this_gene_name, gene_downsampling, data_df)
        model.load(model_path)

        early_stop_callback = EarlyStopping(monitor="loss", min_delta=0.0, patience=patience,mode='min')

        if check_val_every_n_epoch is None:
            # not use early stop
            trainer = pl.Trainer(
                max_epochs=max_epoches, 
                progress_bar_refresh_rate=0, 
                reload_dataloaders_every_n_epochs=1, 
                logger = False,
                enable_checkpointing = False,
                enable_model_summary=False,
                )
        else:
            # use early stop
            trainer = pl.Trainer(
                max_epochs=max_epoches, 
                progress_bar_refresh_rate=0, 
                reload_dataloaders_every_n_epochs=1, 
                logger = False,
                enable_checkpointing = False,
                check_val_every_n_epoch = check_val_every_n_epoch,
                enable_model_summary=False,
                callbacks=[early_stop_callback]
                )

        if max_epoches > 0:
            trainer.fit(model, selected_data)   # train network

        trainer.test(model, selected_data,verbose=False)    # predict
        
        if(model_save_path != None):
            model.save(model_save_path)

        loss_df = model.validation_loss_df
        cellDancer_df = model.test_cellDancer_df

        if norm_u_s:
            cellDancer_df.unsplice=cellDancer_df.unsplice*unsplicemax
            cellDancer_df.splice=cellDancer_df.splice*splicemax
            cellDancer_df.unsplice_predict=cellDancer_df.unsplice_predict*unsplicemax
            cellDancer_df.splice_predict=cellDancer_df.splice_predict*splicemax
            cellDancer_df.beta=cellDancer_df.beta*unsplicemax
            cellDancer_df.gamma=cellDancer_df.gamma*splicemax

        if(model_save_path != None):
            model.save(model_save_path)
        
        header_loss_df=['gene_name','epoch','loss']
        header_cellDancer_df=['cellIndex','gene_name','unsplice','splice','unsplice_predict','splice_predict','alpha','beta','gamma','loss']
        
        loss_df.to_csv(os.path.join(save_path,'TEMP', ('loss'+'_'+this_gene_name+'.csv')),header=header_loss_df,index=False)
        cellDancer_df.to_csv(os.path.join(save_path,'TEMP', ('cellDancer_estimation_'+this_gene_name+'.csv')),header=header_cellDancer_df,index=False)
        
        return None

    except:
        return this_gene_name





def build_datamodule(cell_type_u_s,
                   speed_up,
                   norm_u_s,
                   permutation_ratio, 
                   norm_cell_distribution=False, 
                   gene_list=None,
                   downsample_method='neighbors',
                   n_neighbors_downsample=30,
                   step=(200,200),
                   downsample_target_amount=None):
    
    '''
    set fitting data, data to be predicted, and sampling ratio when fitting
    '''
    step_i=step[0]
    step_j=step[1]
    
    if gene_list is None:
        data_df=cell_type_u_s[['gene_name', 'unsplice','splice','embedding1','embedding2','cellID']]
    else:
        data_df=cell_type_u_s[['gene_name', 'unsplice','splice','embedding1','embedding2','cellID']][cell_type_u_s.gene_name.isin(gene_list)]

    if speed_up:
        _, sampling_ixs, _ = downsampling_embedding(data_df,
                            para=downsample_method,
                            target_amount=downsample_target_amount,
                            step=(step_i,step_j),
                            n_neighbors=n_neighbors_downsample,
                            projection_neighbor_choice='embedding')
        data_df_one_gene=cell_type_u_s[cell_type_u_s['gene_name']==list(gene_list)[0]]
        downsample_cellid=data_df_one_gene.cellID.iloc[sampling_ixs]
        gene_downsampling=data_df[data_df.cellID.isin(downsample_cellid)]

        feed_data = feedData(data_fit = gene_downsampling, data_predict=data_df, permutation_ratio=permutation_ratio,norm_u_s=norm_u_s,norm_cell_distribution=norm_cell_distribution) # default 
    else:
        feed_data = feedData(data_fit = data_df, data_predict=data_df, permutation_ratio=permutation_ratio,norm_u_s=norm_u_s,norm_cell_distribution=norm_cell_distribution) # default 

    return(feed_data)


def velocity(
    cell_type_u_s,
    gene_list=None,
    max_epoches=200, 
    check_val_every_n_epoch=10,
    patience=3,
    learning_rate=0.001,
    dt=0.5,
    n_neighbors=30,
    permutation_ratio=0.125,
    speed_up=True,
    norm_u_s=True,
    norm_cell_distribution=True,
    loss_func='cosine',
    n_jobs=-1,
    save_path=None,
):

    """Velocity estimation for each cell.
        
    Arguments
    ---------
    cell_type_u_s: `pandas.DataFrame`
        Dataframe that contains the unspliced abundance, spliced abundance, embedding space, and cell type information. Columns=['gene_name', 'unsplice', 'splice' ,'cellID' ,'clusters' ,'embedding1' ,'embedding2']
    gene_list: optional, `list` (default: None)
        Gene list for velocity estimation. `None` if to estimate the velocity of all genes.
    max_epoches: optional, `int` (default: 200)
        Stop to update the network once this number of epochs is reached.
    check_val_every_n_epoch: optional, `int` (default: 10)
        Check loss every n train epochs.
    patience: optional, `int` (default: 3)
        Number of checks with no improvement after which training will be stopped.
    dt: optional, `float` (default: 0.5)
        Step size
    permutation_ratio: optional, `float` (default: 0.125)
        Sampling ratio of cells in each epoch when training each gene.
    speed_up: optional, `bool` (default: True)
        `True` if speed up by downsampling cells. `False` if to use all cells to train the model.
    norm_u_s: optional, `bool` (default: True)
        `True` if normalize unsplice (and splice) reads by dividing max value of unspliced (and spliced) reads.
    norm_cell_distribution: optional, `bool` (default: True)
        `True` if the bias of cell distribution is to be removed on embedding space (many cells share the same position of unspliced (and spliced) reads).
    loss_func: optional, `str` (default: `cosine`)
        Currently support `'cosine'`, `'rmse'`, and (`'mix'`, mix_ratio).
    n_jobs: optional, `int` (default: -1)
        The maximum number of concurrently running jobs.
    save_path: optional, `str` (default: 200)
        Path to save the result of velocity estimation.
    Returns
    -------
    loss_df: `pandas.DataFrame`
        The record of loss.
    cellDancer_df: `pandas.DataFrame`
        The result of velocity estimation.
    """

    # set output dir
    datestring = datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S");
    folder_name='cellDancer_velocity_'+datestring

    if save_path is None:
        save_path=os.getcwd()

    try:shutil.rmtree(os.path.join(save_path,folder_name))
    except:os.mkdir(os.path.join(save_path,folder_name))
    save_path=os.path.join(save_path,folder_name)
    print('Using '+save_path+' as the output path.')

    try:shutil.rmtree(os.path.join(save_path,'TEMP'))
    except:os.mkdir(os.path.join(save_path,'TEMP'))
    
    # set gene_list if not given
    if gene_list is None:
        gene_list=list(cell_type_u_s.gene_name.drop_duplicates())
    else:
        cell_type_u_s=cell_type_u_s[cell_type_u_s.gene_name.isin(gene_list)]
        all_gene_name_cell_type_u_s=list(cell_type_u_s.gene_name.drop_duplicates())
        gene_not_in_cell_type_u_s= list(set(gene_list).difference(set(all_gene_name_cell_type_u_s)))
        gene_list=list(list(set(all_gene_name_cell_type_u_s).intersection(set(gene_list))))
        if len(gene_not_in_cell_type_u_s)>0: print(gene_not_in_cell_type_u_s," not in the data cell_type_u_s")

    cell_type_u_s=cell_type_u_s.reset_index(drop=True)
    # buring
    gene_list_buring=[list(cell_type_u_s.gene_name.drop_duplicates())[0]]
    datamodule=build_datamodule(cell_type_u_s,speed_up,norm_u_s,permutation_ratio,norm_cell_distribution,gene_list=gene_list_buring)

    result = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_train_thread)(
            datamodule = datamodule,
            data_indices=[data_index], 
            max_epoches=max_epoches,
            check_val_every_n_epoch=check_val_every_n_epoch,
            patience=patience,
            learning_rate=learning_rate,
            n_neighbors=n_neighbors,
            dt=dt,
            loss_func=loss_func,
            save_path=save_path,
            norm_u_s=norm_u_s)
        for data_index in range(0,len(gene_list_buring)))

    # clean directory
    shutil.rmtree(os.path.join(save_path,'TEMP'))
    os.mkdir(os.path.join(save_path,'TEMP'))
    
    data_len = len(gene_list)
    
    id_ranges=list()
    if n_jobs==-1:
        interval=os.cpu_count()
    else:
        interval=n_jobs
    for i in range(0,data_len,interval):
        idx_start=i
        if data_len<i+interval:
            idx_end=data_len
        else:
            idx_end=i+interval
        id_ranges.append((idx_start,idx_end))


    print('Arranging genes for parallel job.')
    if len(id_ranges)==1:
        if id_ranges==1:
            print(data_len,' gene was arranged to ',len(id_ranges),' portion.')
        else:
            print(data_len,' genes were arranged to ',len(id_ranges),' portion.')
    else: 
        print(data_len,' genes were arranged to ',len(id_ranges),' portions.')
    
    unpredicted_gene_lst=list()
    for id_range in tqdm(id_ranges,desc="Velocity Estimation", total=len(id_ranges),position=1,leave=False, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'):
        gene_list_batch=gene_list[id_range[0]:id_range[1]]
        datamodule=build_datamodule(cell_type_u_s,speed_up,norm_u_s,permutation_ratio,norm_cell_distribution,gene_list=gene_list_batch)

        result = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_train_thread)(
            datamodule = datamodule,
            data_indices=[data_index], 
            max_epoches=max_epoches,
            check_val_every_n_epoch=check_val_every_n_epoch,
            n_neighbors=n_neighbors,
            dt=dt,
            loss_func=loss_func,
            learning_rate=learning_rate,
            patience=patience,
            save_path=save_path,
            norm_u_s=norm_u_s)
            for data_index in range(0,len(gene_list_batch)))

        # unpredicted gene list
        gene_name_lst=[x for x in result if x is not None]
        for i in gene_name_lst:
            unpredicted_gene_lst.append(i)
    if len(unpredicted_gene_lst)!=0:
        not_pred_err='Not predicted gene list:'+str(unpredicted_gene_lst)+'. Try visualizing the unspliced and spliced columns of the gene(s) to check the quality.'
        logger_cd.error(not_pred_err)

    # summarize
    cellDancer_df = os.path.join(save_path,'TEMP', "cellDancer_estimation*.csv")
    cellDancer_df_files = glob.glob(cellDancer_df)
    loss_df = os.path.join(save_path, 'TEMP',"loss*.csv")
    loss_df_files = glob.glob(loss_df)

    def combine_csv(save_path,files):
        with open(save_path,"wb") as fout:
            # first file:
            with open(files[0], "rb") as f:
                fout.write(f.read())
            # the rest:    
            for filepath in files[1:]:
                with open(filepath, "rb") as f:
                    next(f)
                    fout.write(f.read())
        return(pd.read_csv(save_path))

    if len(cellDancer_df_files)==0:
        # if no gene predicted
        logger_cd.error('None of the genes were predicted. Try visualizing the unspliced and spliced columns of the gene(s) to check the quality.')
        return None, None
    else:
        cellDancer_df=combine_csv(os.path.join(save_path,"cellDancer_estimation.csv"),cellDancer_df_files)
        loss_df=combine_csv(os.path.join(save_path,"cellDancer_estimation.csv"),loss_df_files)

        shutil.rmtree(os.path.join(save_path,'TEMP'))

        cellDancer_df.sort_values(by = ['gene_name', 'cellIndex'], ascending = [True, True])
        onegene=cell_type_u_s[cell_type_u_s.gene_name==cell_type_u_s.gene_name[0]]
        embedding_info=onegene[['cellID','clusters','embedding1','embedding2']]
        gene_amt=len(cellDancer_df.gene_name.drop_duplicates())
        embedding_col=pd.concat([embedding_info]*gene_amt)
        embedding_col.index=cellDancer_df.index
        cellDancer_df=pd.concat([cellDancer_df,embedding_col],axis=1)
        cellDancer_df.to_csv(os.path.join(save_path, ('cellDancer_estimation.csv')),index=False)

        loss_df.to_csv(os.path.join(save_path, ('loss.csv')),index=False)

        return loss_df, cellDancer_df

    
def select_initial_net(gene, gene_downsampling, data_df):
    '''
    check if right top conner has cells
    circle.pt is the model for single kinetic
    branch.pt is multiple kinetic
    '''
    gene_u_s = gene_downsampling[gene_downsampling.gene_name==gene]
    gene_u_s_full = data_df[data_df.gene_name==gene]
    
    s_max=np.max(gene_u_s.splice)
    u_max = np.max(gene_u_s.unsplice)
    s_max_90per = 0.9*s_max
    u_max_90per = 0.9*u_max
    
    gene_u_s_full['position'] = 'position_cells'
    gene_u_s_full.loc[(gene_u_s_full.splice>s_max_90per) & (gene_u_s_full.unsplice>u_max_90per), 'position'] = 'cells_corner'

    if gene_u_s_full.loc[gene_u_s_full['position']=='cells_corner'].shape[0]>0.001*gene_u_s_full.shape[0]:
        # model in circle shape
        model_path=pkg_resources.resource_stream(__name__,os.path.join('model', 'circle.pt')).name
    else:
        # model in seperated branch shape
        model_path=pkg_resources.resource_stream(__name__,os.path.join('model', 'branch.pt')).name
    return(model_path)