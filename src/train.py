import os

import numpy as np
import torch
from torch import nn
from torch.optim.sgd import SGD
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data import DatasetMusic
from src.network import AudioNetwork


def test_triplet_training():
    """Performs a test of the training pipeline."""

    # Configure the network
    network = AudioNetwork()
    network.train()

    # Setup optimizer
    optimizer = SGD(network.parameters(), lr=0.001)

    # Setup the loss function
    triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)

    waveform_anc = torch.ones(4, 2, 64, 2584, requires_grad=True)
    waveform_pos = torch.zeros(4, 2, 64, 2584, requires_grad=True)
    waveform_neg = torch.randn(4, 2, 64, 2584, requires_grad=True)

    # Iterate over epochs
    for epoch_idx in range(0, 3):

        # Perform training
        for iter_idx in range(0, 3):

            optimizer.zero_grad()

            anchor__ = network(waveform_anc)
            positive = network(waveform_pos)
            negative = network(waveform_neg)

            # citation: https://pytorch.org/docs/stable/generated/torch.nn.TripletMarginLoss.html
            output = triplet_loss(anchor__, positive, negative)

            print(f"output loss={output}")

            output.backward()

            optimizer.step()


def triplet_training():
    """Performs the training process."""

    # Variables
    model_save_path = "save/audio_network.pkl"

    # Setup the dataset
    dataset_train = DatasetMusic(train=True, spectrogram=True)
    dataset_valid = DatasetMusic(train=False, spectrogram=True)

    # Setup dataloaders
    dataloader_train = DataLoader(
        dataset_train, batch_size=16, num_workers=4, prefetch_factor=1
    )
    dataloader_valid = DataLoader(
        dataset_valid, batch_size=16, num_workers=4, prefetch_factor=1
    )

    # Configure the network
    network = AudioNetwork()

    if os.path.exists(model_save_path):
        network.load_state_dict(torch.load(model_save_path))

    # Setup optimizer
    optimizer = SGD(network.parameters(), lr=0.001)

    # Setup the loss function
    triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)

    # Iterate over epochs
    for epoch_idx in tqdm(range(0, 10), desc="Epoch", colour="green"):

        # Perform training
        network.train()
        for sample in tqdm(dataloader_train, desc="Iteration", colour="blue"):

            waveform_anc = sample[0]
            waveform_pos = sample[1]
            waveform_neg = sample[2]

            optimizer.zero_grad()

            anchor__ = network(waveform_anc)
            positive = network(waveform_pos)
            negative = network(waveform_neg)

            # citation: https://pytorch.org/docs/stable/generated/torch.nn.TripletMarginLoss.html
            output = triplet_loss(anchor__, positive, negative)
            output.backward()

            optimizer.step()

        # Perform validation and calculate loss
        network.eval()
        max_validation_loss = 100
        losses = []
        for sample in tqdm(dataloader_valid, desc="Validating", colour="red"):

            waveform_anc = sample[0]
            waveform_pos = sample[1]
            waveform_neg = sample[2]

            with torch.no_grad():
                anchor__ = network(waveform_anc)
                positive = network(waveform_pos)
                negative = network(waveform_neg)

                output = triplet_loss(anchor__, positive, negative)

                losses.append(output)

        if np.mean(losses) < max_validation_loss:
            max_validation_loss = np.mean(losses)
            torch.save(network.state_dict(), model_save_path)
            print("Saving best model...")

        print(f"Average Validation Loss: {np.mean(losses)}")


if __name__ == "__main__":

    # Test the pipeline
    # test_triplet_training()

    # Run the pipeline with the real data
    triplet_training()
