import os
from typing import Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split


SEED = 76
torch.manual_seed(SEED)
np.random.seed(SEED)


DATASET_ROOT = os.path.join(os.getcwd(), "data")


class TransformedDataset(Dataset):
    def __init__(self, dataset: Dataset, transform: transforms.Compose):
        self.dataset = dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        image, label = self.dataset[index]
        return self.transform(image), label


def get_transforms() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )


def get_eval_transforms() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )


def load_cifar10() -> Tuple[Dataset, Dataset, Dataset]:
    raw_train_set = datasets.CIFAR10(root=DATASET_ROOT, train=True, download=True)
    test_set = datasets.CIFAR10(root=DATASET_ROOT, train=False, download=True, transform=get_eval_transforms())

    labels = np.array(raw_train_set.targets)
    train_idx, val_idx = train_test_split(
        np.arange(len(raw_train_set)),
        test_size=0.2,
        stratify=labels,
        random_state=SEED,
    )

    train_set = TransformedDataset(raw_train_set, get_transforms())
    val_set = TransformedDataset(raw_train_set, get_eval_transforms())
    train_subset = Subset(train_set, train_idx)
    val_subset = Subset(val_set, val_idx)
    return train_subset, val_subset, test_set


def print_dataset_summary(train_set: Dataset, val_set: Dataset, test_set: Dataset) -> None:
    print(f"Train samples: {len(train_set)}")
    print(f"Validation samples: {len(val_set)}")
    print(f"Test samples: {len(test_set)}")
    print(f"Train input shape: {train_set[0][0].shape}")
    print(f"Train label example: {train_set[0][1]}")


def show_sample_grid(dataset: Dataset, rows: int = 3, cols: int = 4, title: str = "Samples CIFAR-10") -> None:
    fig, axes = plt.subplots(rows, cols, figsize=(12, 8))
    fig.suptitle(title, fontsize=14)
    class_names = [
        "airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"
    ]

    for ax in axes.flat:
        ax.axis("off")

    for idx in range(rows * cols):
        image, label = dataset[idx]
        image = image.permute(1, 2, 0).numpy()
        image = (image * np.array((0.2023, 0.1994, 0.2010))) + np.array((0.4914, 0.4822, 0.4465))
        image = np.clip(image, 0, 1)
        row, col = divmod(idx, cols)
        axes[row, col].imshow(image)
        axes[row, col].set_title(class_names[label])

    plt.tight_layout()
    plt.savefig(os.path.join(os.getcwd(), "outputs", "dataset_samples.png"), dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    train_set, val_set, test_set = load_cifar10()
    print_dataset_summary(train_set, val_set, test_set)
    show_sample_grid(train_set, rows=3, cols=4, title="Muestra del dataset CIFAR-10")
    print("Grid saved to outputs/dataset_samples.png")
