# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 23:10:49 2020

@author: del
"""

import time
import os
import json
from sys import platform
import argparse
import numpy as np
import torch

from torch.utils.data import DataLoader
from data import QQPDataset
from model import ESIM
from utils import correct_predictions

def test(model, dataloader):
    """
    Test the accuracy of a model on some labelled test dataset.
    Args:
        model: The torch module on which testing must be performed.
        dataloader: A DataLoader object to iterate over some dataset.
    Returns:
        batch_time: The average time to predict the classes of a batch.
        total_time: The total time to process the whole dataset.
        accuracy: The accuracy of the model on the input data.
    """
    # Switch the model to eval mode.
    model.eval()
    device = model.device

    time_start = time.time()
    batch_time = 0.0
    accuracy = 0.0

    # Deactivate autograd for evaluation.
    with torch.no_grad():
        for batch in dataloader:
            batch_start = time.time()

            # Move input and output data to the GPU if one is used.
            q1 = batch["q1"].to(device)
            q1_lengths = batch["q1_length"].to(device)
            q2 = batch["q2"].to(device)
            q2_lengths = batch["q2_length"].to(device)
            labels = batch["label"].to(device)

            _, probs = model(q1, q1_lengths, q2, q2_lengths)

            accuracy += correct_predictions(probs, labels)
            batch_time += time.time() - batch_start

    batch_time /= len(dataloader)
    total_time = time.time() - time_start
    accuracy /= (len(dataloader.dataset))

    return batch_time, total_time, accuracy


def main(test_q1_file, test_q2_file, test_labels_file,
         pretrained_file, 
         gpu_index=0, 
         batch_size=64):
    """
    Test the ESIM model with pretrained weights on some dataset.
    Args:
        test_file: The path to a file containing preprocessed NLI data.
        pretrained_file: The path to a checkpoint produced by the
            'train_model' script.
        vocab_size: The number of words in the vocabulary of the model
            being tested.
        embedding_dim: The size of the embeddings in the model.
        hidden_size: The size of the hidden layers in the model. Must match
            the size used during training. Defaults to 300.
        num_classes: The number of classes in the output of the model. Must
            match the value used during training. Defaults to 3.
        batch_size: The size of the batches used for testing. Defaults to 32.
    """
    device = torch.device("cuda:{}".format(gpu_index) if torch.cuda.is_available() else "cpu")

    print(20 * "=", " Preparing for testing ", 20 * "=")
    
    if platform == "linux" or platform == "linux2":
        checkpoint = torch.load(pretrained_file)
    else:
        checkpoint = torch.load(pretrained_file, map_location="cuda:0")
    
    # Retrieving model parameters from checkpoint.
    vocab_size = checkpoint["model"]["word_embedding.weight"].size(0)
    embedding_dim = checkpoint["model"]['word_embedding.weight'].size(1)
    hidden_size = checkpoint["model"]["projection.0.weight"].size(0)
    num_classes = checkpoint["model"]["classification.6.weight"].size(0)

    print("\t* Loading test data...")
    test_q1 = np.load(test_q1_file)
    test_q2 = np.load(test_q2_file)
    test_labels = np.load(test_labels_file)
#    test_labels = label_transformer(test_labels)
    
    test_data = {"q1": test_q1,
                 "q2": test_q2,
                 "labels": test_labels}
    
    test_data = QQPDataset(test_data)
    test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size)

    print("\t* Building model...")
    model = ESIM(vocab_size,
                 embedding_dim,
                 hidden_size,
                 num_classes=num_classes,
                 device=device).to(device)

    model.load_state_dict(checkpoint["model"])

    print(20 * "=",
          " Testing ESIM model on device: {} ".format(device),
          20 * "=")
    batch_time, total_time, accuracy = test(model, test_loader)

    print()
    print("-> Average batch processing time: {:.4f}s, total test time: {:.4f}s, accuracy: {:.4f}%".format(batch_time, total_time, (accuracy*100)))
    print()


if __name__ == "__main__":
    default_config = "./config.json"
    parser = argparse.ArgumentParser(description="Test the ESIM model on some dataset")
    parser.add_argument("--config",
                        default=default_config,
                        help="Path to a json configuration file")
    parser.add_argument("checkpoint",
                        help="Path to a checkpoint with a pretrained model")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size to use during testing")
    parser.add_argument("--gpu",
                        default=0,
                        help="which cuda device to use")
    args = parser.parse_args()
    
    script_dir = os.path.dirname(os.path.realpath(__file__))
    
    if args.config == default_config:
        config_path = os.path.join(script_dir, args.config)
    else:
        config_path = args.config

    with open(os.path.normpath(config_path), 'r') as config_file:
        config = json.load(config_file)

    main(os.path.normpath(os.path.join(script_dir, config["test_q1_data"])),
         os.path.normpath(os.path.join(script_dir, config["test_q2_data"])),
         os.path.normpath(os.path.join(script_dir, config["test_labels_data"])),
         args.checkpoint,
         args.gpu,
         args.batch_size)