import sys
from functools import partial

import wandb
import os
from torch.utils.data import DataLoader, Subset
import numpy as np

import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from omegaconf import OmegaConf

import torch
from sklearn.model_selection import KFold
from thermompnn.parsers import get_v2_dataset
from thermompnn.trainer.v2_trainer import TransferModelPLv2, TransferModelPLv2Siamese
from thermompnn.datasets.v2_datasets import tied_featurize_mut


def collate_fn(batch, side_chains):
    return tied_featurize_mut(batch, side_chains=side_chains)

def parse_cfg(cfg):
    """
    Parse configuration scheme and set default arguments as needed
    """
    cfg.project = cfg.get('project', None)
    cfg.name = cfg.get('name', 'test')

    # data config
    cfg.data = cfg.get('data', {})
    cfg.data.mut_types = cfg.data.get('mut_types', ['single'])
    cfg.data.splits = cfg.data.get('splits', ['train', 'val'])
    cfg.data.side_chains = cfg.data.get('side_chains', False)
    cfg.data.refresh_every = cfg.data.get('refresh_every', 0)
    cfg.data.weight = cfg.data.get('weight', False)
    cfg.data.range = cfg.data.get('range', None)

    # training config
    cfg.training = cfg.get('training', {})
    cfg.training.num_workers = cfg.training.get('num_workers', 0)
    cfg.training.batch_size = cfg.training.get('batch_size', 256)
    cfg.training.epochs = cfg.training.get('epochs', 100)
    cfg.training.batch_fraction = cfg.training.get('batch_fraction', 1.0)
    cfg.training.shuffle = cfg.training.get('shuffle', True)

    cfg.training.learn_rate = cfg.training.get('learn_rate', 0.0001)
    cfg.training.mpnn_learn_rate = cfg.training.get('mpnn_learn_rate', None)
    cfg.training.lr_schedule = cfg.training.get('lr_schedule', True)

    # model config
    cfg.model = cfg.get('model', {})
    cfg.model.hidden_dims = cfg.model.get('hidden_dims', [64, 32])
    cfg.model.subtract_mut = cfg.model.get('subtract_mut', True)
    cfg.model.single_target = cfg.model.get('single_target', False)
    cfg.model.num_final_layers = cfg.model.get('num_final_layers', 2)
    cfg.model.freeze_weights = cfg.model.get('freeze_weights', True)
    cfg.model.load_pretrained = cfg.model.get('load_pretrained', True)
    cfg.model.lightattn = cfg.model.get('lightattn', True)
    cfg.model.mutant_embedding = cfg.model.get('mutant_embedding', False)
    cfg.model.alpha = cfg.model.get('alpha', 1.0)
    cfg.model.beta = cfg.model.get('beta', 1.0)
    
    # double mutant model options
    cfg.model.dist = cfg.model.get('dist', False)
    cfg.model.edges = cfg.model.get('edges', False)
    cfg.model.aggregation = cfg.model.get('aggregation', None)
    cfg.model.dropout = cfg.model.get('dropout', None)

    # side chain model options
    cfg.model.side_chain_module = cfg.model.get('side_chain_module', False)
    cfg.model.action_centers = cfg.model.get('action_centers', None)

    return cfg


def train(cfg):
    print('Configuration:\n', cfg)

    cfg = parse_cfg(cfg)

    if cfg.project is not None:
        wandb.init(project=cfg.project, name=cfg.name)

    train_dataset = get_v2_dataset(cfg)

    if cfg.model.aggregation == 'siamese':
        model_pl = TransferModelPLv2Siamese(cfg)
    else:
        model_pl = TransferModelPLv2(cfg)
        checkpoint = torch.load("DetergentMPNN/model_weights/ThermoMPNN-ens1.ckpt", map_location="cpu")
        model_pl.load_state_dict(checkpoint['state_dict'], strict=True)
        #are_identical = all(torch.equal(model_pl.state_dict()[key], checkpoint['state_dict'][key]) for key in model_pl.state_dict())

    for name, param in model_pl.named_parameters():
        print(f"{name}: requires_grad = {param.requires_grad}")
    # additional params, logging, checkpoints for training
    '''filename = cfg.name + '_{epoch:02d}_{val_ddG_spearman:.02}'
    monitor = f'val_ddG_spearman'
    
    current_location = os.path.dirname(os.path.realpath(__file__))
    checkpath = os.path.join(current_location, 'checkpoints/')
    if not os.path.isdir(checkpath):
        os.mkdir(checkpath)

    checkpoint_callback = ModelCheckpoint(monitor=monitor, mode='max', dirpath=checkpath, filename=filename)
    logger = WandbLogger(project=cfg.project, name="test", log_model=False) if cfg.project is not None else None'''
    n_steps = 100
    
    csv_logger = CSVLogger("/content/DetergentMPNN/logs", name="training_metrics")

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    dataset_size = len(train_dataset)
    indices = np.arange(dataset_size)

    for fold, (train_idx, val_idx) in enumerate(kf.split(indices)):
      print(f"Fold {fold + 1} / 5")
      print(train_idx)
      print(val_idx)

      trainer = pl.Trainer(max_epochs=cfg.training.epochs,
                        logger=csv_logger,
                        log_every_n_steps=n_steps,
                        accelerator=cfg.platform.accel, 
                        devices=1, 
                        limit_train_batches=cfg.training.batch_fraction, 
    )
    
      train_loader = DataLoader(Subset(train_dataset, train_idx), 
                                  collate_fn= partial(collate_fn, side_chains=cfg.data.side_chains),
                                  shuffle=cfg.training.shuffle, 
                                  num_workers=cfg.training.num_workers, 
                                  batch_size=cfg.training.batch_size)
      val_loader = DataLoader(Subset(train_dataset, val_idx), 
                                  collate_fn=partial(collate_fn, side_chains=cfg.data.side_chains),
                                  shuffle=False, 
                                  num_workers=cfg.training.num_workers, 
                                  batch_size=cfg.training.batch_size)

      
      trainer.fit(model_pl, train_loader, val_loader) #, ckpt_path=cfg.training.ckpt)
    torch.save(model_pl.state_dict(), "detergent_mpnn_weights.ckpt")


if __name__ == "__main__":
    # config.yaml and local.yaml files are combined to assemble all runtime arguments
    print(len(sys.argv))
    if len(sys.argv) != 3:
        raise ValueError("Need to specify exactly two config files.")
    
    cfg = OmegaConf.merge(OmegaConf.load(sys.argv[1]), OmegaConf.load(sys.argv[2]))
    train(cfg)
