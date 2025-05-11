from thermompnn.datasets.v2_datasets import MegaScaleDatasetv2, DetergentDataset

def get_v2_dataset(cfg):
    query = cfg.data.dataset.lower()
    splits = cfg.data.splits
    if query.startswith('megascale'):
        return MegaScaleDatasetv2(cfg, splits[0])
    elif query.startswith('detergent'):
        return DetergentDataset(cfg)
    else:
        raise ValueError("Invalid training dataset '%s' selected!" % query)
