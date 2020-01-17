# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 18:12:55 2020

@author: del
"""
import torch
from torch.utils.data import Dataset

class QQPDataset(Dataset):
    """
    Dataset class for QQP datasets.
    """

    def __init__(self,
                 data,
                 padding_idx=0,
                 max_length=25):
        """
        Args:
            data: A dictionary containing the preprocessed premises,
                hypotheses and labels of some dataset.
            padding_idx: An integer indicating the index being used for the
                padding token in the preprocessed data. Defaults to 0.
            max_premise_length: An integer indicating the maximum length
                accepted for the sequences in the premises. If set to None,
                the length of the longest premise in 'data' is used.
                Defaults to None.
            max_hypothesis_length: An integer indicating the maximum length
                accepted for the sequences in the hypotheses. If set to None,
                the length of the longest hypothesis in 'data' is used.
                Defaults to None.
        """
        self.q1_lengths = [(seq != padding_idx).sum() for seq in data["q1"]]
        self.q2_lengths = [(seq != padding_idx).sum() for seq in data["q2"]]
        self.max_length = max_length
        self.num_sequences = len(data["q1"])

        self.data = {"q1": torch.ones((self.num_sequences,
                                       self.max_length),
                                       dtype=torch.long) * padding_idx,
                     "q2": torch.ones((self.num_sequences,
                                       self.max_length),
                                       dtype=torch.long) * padding_idx,
                     "labels": torch.tensor(data["labels"], dtype=torch.long)}
        
        for (i, q1), q1_len in (zip(enumerate(data["q1"]), self.q1_lengths)):
            end = q1_len
            self.data["q1"][i][:end] = torch.tensor(q1[:end])

            q2 = data["q2"][i]
            end = (q2 != padding_idx).sum()
            self.data["q2"][i][:end] = torch.tensor(q2[:end])

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, index):
        return {"q1": self.data["q1"][index],
                "q1_length": min(self.q1_lengths[index],
                                 self.max_length),
                "q2": self.data["q2"][index],
                "q2_length": min(self.q2_lengths[index],
                                 self.max_length),
                "label": self.data["labels"][index]}