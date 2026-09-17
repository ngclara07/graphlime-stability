# src/data.py

from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures


def load_cora(root: str = "data"):
    """
    Load the Cora citation-network dataset using the standard
    Planetoid split and row-normalize node features.

    Parameters
    ----------
    root : str
        Directory in which PyG stores the downloaded dataset.

    Returns
    -------
    dataset
        PyTorch Geometric Planetoid dataset object.
    data
        Cora graph represented as a PyG Data object.
    """
    dataset = Planetoid(
        root=root,
        name="Cora",
        transform=NormalizeFeatures(),
    )

    data = dataset[0]

    return dataset, data
