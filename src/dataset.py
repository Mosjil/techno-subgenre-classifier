import torch
from torch.utils.data import Dataset
from utils.utils import parse_labels
import pandas as pd
import numpy as np
import ast

# TODO : Data Augmentation ?
class TechnoDataset(Dataset):
    def __init__(self, csv_path, class_list, transform=None):
        """
        Args:
            csv_path (str): chemin vers le CSV (path, sous-genres)
            class_list (list[str]): liste globale des sous-genres possibles
            transform (callable, optional): transformations à appliquer au mel-spectrogram
        """
        self.df = pd.read_csv(csv_path)

        self.class_list = class_list
        self.num_classes = len(class_list)

        self.transform = transform

    def __len__(self):
        """Total number of samples"""
        return len(self.df)

    def __getitem__(self, idx):
        """Load a sample"""
        row = self.df.iloc[idx]

        mel = np.load(row["path_spec"])
        mel = torch.tensor(mel, dtype=torch.float32)
        mel = mel.unsqueeze(0)  # (1, n_mels, T)

        genres = parse_labels(row["subgenres"])
        # genres = ast.literal_eval(row["subgenres"])

        label = torch.zeros(self.num_classes, dtype=torch.float32)
        for g in genres:
            if g in self.class_list:
                label[self.class_list.index(g)] = 1.0

        if self.transform:
            mel = self.transform(mel)

        return mel, label


# classes = ["Acid Techno", "Hard Techno", "Melodic Techno", "Minimal Techno", "Detroit Techno"]
#
# dataset = GenreSpecDataset("data/segments.csv", class_list=classes)
# mel, label = dataset[0]
#
# print(mel.shape)   # torch.Size([1, 256, T])
# print(label)       # tensor([0., 0., 1., 1., 0.])
