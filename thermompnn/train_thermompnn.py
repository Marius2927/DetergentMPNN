import sys
from functools import partial
import pandas
import wandb
import os
import shutil
from collections import defaultdict

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:512"
from pandas.core.common import flatten
from torch.utils.data import DataLoader, Subset
import numpy as np

import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from omegaconf import OmegaConf

sys.path.append('~/DetergentMPNN')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from sklearn.model_selection import KFold, BaseCrossValidator
from thermompnn.parsers import get_v2_dataset
from thermompnn.trainer.v2_trainer import TransferModelPLv2, TransferModelPLv2Siamese
from thermompnn.datasets.v2_datasets import tied_featurize_mut

torch.cuda.empty_cache()

class BackboneKFold(BaseCrossValidator):
    def __init__(self, backbone_ids):
        self.backbone_ids = np.array(backbone_ids)
        self.unique_backbones = np.unique(self.backbone_ids)
        self.n_splits = len(self.unique_backbones)
        self.type = "backbone_fold"
        # df = pandas.DataFrame(self.unique_backbones, columns=['id'])
        # df.to_csv('data/pdb_ids.csv')

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def split(self, X, y=None, groups=None):
        for backbone in self.unique_backbones:
            test_indices = np.where(self.backbone_ids == backbone)[0]
            train_indices = np.where(self.backbone_ids != backbone)[0]
            yield train_indices, test_indices

class SiteKFold(BaseCrossValidator):
    def __init__(self, site_ids, n_splits=5):
        self.site_ids = np.array(site_ids)
        self.n_splits = n_splits
        self.unique_sites = np.unique(self.site_ids)
        self.type = "site_fold"

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def split(self, X, y=None, groups=None):
        X = np.array(X)
        assert len(X) == len(self.site_ids), "X and site_ids must be the same length"

        site_to_indices = defaultdict(list)
        for idx, site in enumerate(self.site_ids):
            site_to_indices[site].append(idx)

        # Round-robin assignment of sites to folds
        fold_buckets = [[] for _ in range(self.n_splits)]
        for i, site in enumerate(self.unique_sites):
            fold = i % self.n_splits
            fold_buckets[fold].extend(site_to_indices[site])

        # Yield train/val indices
        for i in range(self.n_splits):
            val_indices = np.array(fold_buckets[i])
            train_indices = np.array([idx for j, bucket in enumerate(fold_buckets) if j != i for idx in bucket])
            yield train_indices, val_indices

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
    cfg.training.cv_type = cfg.training.get('cv')

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

    torch.set_float32_matmul_precision('medium')


    #logger = WandbLogger(project=cfg.project, name="test", log_model=False) if cfg.project is not None else None
    n_steps = 100
    pdb_ids = train_dataset.df['pdb']
    site_ids = [mut_str[1:-1] for mut_str in train_dataset.df['mutant']]

    if cfg.training.cv_type == 'backbone':
        cv = BackboneKFold(pdb_ids)
        type = cv.type
    elif cfg.training.cv_type == 'site':
        cv = SiteKFold(site_ids)
        type = cv.type
    elif cfg.training.cv_type == 'random':
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        type='random'

    current_location = os.path.dirname(os.path.realpath(__file__))
    checkpath = os.path.join(current_location, 'checkpoints/'+type)
    if not os.path.isdir(checkpath):
        os.mkdir(checkpath)

    dataset_size = len(train_dataset)
    indices = np.arange(dataset_size)
    print("Using device:", torch.cuda.current_device(), torch.cuda.get_device_name(torch.cuda.current_device()))

    best_score = -float('inf')
    best_ckpt_path = None

    all_preds = []
    all_targets = []

    for fold, (train_idx, val_idx) in enumerate(cv.split(indices)):
        print(f"Fold {fold + 1} / {cv.n_splits}")

        if cfg.model.aggregation == 'siamese':
            model_pl = TransferModelPLv2Siamese(cfg)
        else:
            model_pl = TransferModelPLv2(cfg)
            checkpoint = torch.load("model_weights/ThermoMPNN-ens1.ckpt", map_location="cpu")
            model_pl.load_state_dict(checkpoint['state_dict'], strict=True)

        for name, param in model_pl.named_parameters():
            print(f"{name}: requires_grad = {param.requires_grad}")
        # additional params, logging, checkpoints for training
        filename = cfg.name + '_{epoch:02d}_{val_ddG_spearman:.02}'
        monitor = f'val_ddG_spearman'

        fold_ckpt_path = os.path.join(checkpath, f"fold{fold + 1}")
        os.makedirs(fold_ckpt_path, exist_ok=True)
        checkpoint_callback = ModelCheckpoint(
            monitor=monitor,
            mode='max',
            dirpath=fold_ckpt_path,
            filename=f'{cfg.name}_fold{fold + 1}_' + '{epoch:02d}_{val_ddG_spearman:.2f}'
        )

        csv_logger = CSVLogger("logs/" + type, name=f"training_fold{fold + 1}")

        trainer = pl.Trainer(
            callbacks=[checkpoint_callback],
            max_epochs=cfg.training.epochs,
            logger=csv_logger,
            log_every_n_steps=n_steps,
            accelerator=cfg.platform.accel,
            devices=1,
            limit_train_batches=cfg.training.batch_fraction
        )

        train_loader = DataLoader(
            Subset(train_dataset, train_idx),
            collate_fn=partial(collate_fn, side_chains=cfg.data.side_chains),
            shuffle=cfg.training.shuffle,
            num_workers=cfg.training.num_workers,
            batch_size=cfg.training.batch_size
        )

        val_loader = DataLoader(
            Subset(train_dataset, val_idx),
            collate_fn=partial(collate_fn, side_chains=cfg.data.side_chains),
            shuffle=False,
            num_workers=cfg.training.num_workers,
            batch_size=cfg.training.batch_size
        )

        trainer.fit(model_pl, train_loader, val_loader)

        ckpt_path = checkpoint_callback.best_model_path
        ckpt_score = checkpoint_callback.best_model_score.item()

        print(f"Best checkpoint for fold {fold + 1}: {ckpt_path} (score = {ckpt_score:.4f})")

        model_pl.eval()
        model_pl.freeze()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_pl = model_pl.to(device)

        # Collect actual and predicted ddG values
        fold_preds = []
        fold_targets = []

        for batch in val_loader:
            batch = [b.to(device) if isinstance(b, torch.Tensor) else b for b in batch]
            preds, ddgs = model_pl.predict_ddg_batch(batch)

            fold_preds.extend(preds.tolist())
            fold_targets.extend(ddgs.tolist())

        all_preds.extend(fold_preds)
        all_targets.extend(fold_targets)

        if ckpt_score > best_score:
            best_score = ckpt_score
            best_ckpt_path = ckpt_path

        trainer.fit(model_pl, train_loader, val_loader)

    df = pandas.DataFrame({'preds' : flatten(all_preds), 'targets' : flatten(all_targets)})
    df.to_csv('data/detergent_'+type+'_performance.csv')
    print(f"Best overall model from fold checkpoint: {best_ckpt_path}")
    shutil.copyfile(best_ckpt_path, "detergent_mpnn_weights_best_fold.ckpt")
    os.rename(best_ckpt_path, "thermompnn/checkpoints/" + type + "/detergent_mpnn_weights_best_" + type + ".ckpt")


if __name__ == "__main__":
    # config.yaml and local.yaml files are combined to assemble all runtime arguments
    if len(sys.argv) != 3:
        raise ValueError("Need to specify exactly two config files.")
    
    cfg = OmegaConf.merge(OmegaConf.load(sys.argv[1]), OmegaConf.load(sys.argv[2]))
    train(cfg)
