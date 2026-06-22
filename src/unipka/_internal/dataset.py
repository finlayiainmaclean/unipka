import numpy as np
from torch.utils.data import Dataset

UnimolBatch = dict[str, np.ndarray]


class MolDataset(Dataset):
    """
    A :class:`MolDataset` class is responsible for interface of molecular dataset.
    """

    def __init__(
        self, data: list[UnimolBatch], label: np.ndarray | None = None
    ) -> None:
        self.data = data
        self.label = label if label is not None else np.zeros((len(data), 1))

    def __getitem__(self, idx: int) -> tuple[UnimolBatch, np.ndarray]:
        return self.data[idx], self.label[idx]

    def __len__(self) -> int:
        return len(self.data)
