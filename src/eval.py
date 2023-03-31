import numpy as np
import pandas as pd
import torch
from sklearn.metrics import ConfusionMatrixDisplay
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data import DatasetMusic
from src.data import DatasetMusicTest
from src.network import AudioNetwork


def get_anchor_embeddings(network: torch.nn.Module, anchors_waveform: dict) -> dict:

    # Get the embeddings for each anchor
    anchors_embedding = {}
    for name, waveform in anchors_waveform.items():

        # Perform inferencing to get the embedding
        with torch.no_grad():
            output = network(waveform.unsqueeze(0))
            anchors_embedding[name] = output.detach().numpy()

    return anchors_embedding


def plot_confusion_matrix():

    # Variables
    model_save_path = "save/audio_network.pkl"

    # Create the network
    network = AudioNetwork()
    network.load_state_dict(torch.load(model_save_path))

    # Create the dataloader for validation
    dataset_valid = DatasetMusic(train=False, spectrogram=True)
    dataloader_valid = DataLoader(dataset_valid, batch_size=1)

    # Get the anchor waveforms
    anchors_waveforms = dataset_valid.get_anchors()
    anchors_embedding = get_anchor_embeddings(network, anchors_waveforms)

    y_true = []
    y_pred = []

    for sample in tqdm(dataloader_valid, desc="Validating", colour="red"):

        # Waveforms
        waveform_pos = sample[1]

        # Labels
        label_pos = sample[4][0]

        # Generate prediction
        prediction = network(waveform_pos)

        # Need to compare the prediction versus all anchors
        association = None
        max_dist = 1000
        for key, val in anchors_embedding.items():

            # We are going to use Euclidean distance to figure out what to associate the embedding with
            dist = np.linalg.norm(prediction.detach().numpy() - val)
            if dist < max_dist:
                max_dist = dist
                association = key

        # Check if dist is out of bounds
        if max_dist > 5:
            association = "Other"

        # Assign confusion matrix values
        y_true.append(label_pos)
        y_pred.append(association)

    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        labels=["Beethoven", "Schubert", "Bach", "Brahms", "Other"],
    )

    print("Stop here to view the confusion matrix!")


def infer_test_data():

    # Variables
    model_save_path = "save/audio_network.pkl"

    # Create the network
    network = AudioNetwork()
    network.load_state_dict(torch.load(model_save_path))
    network.eval()

    # Create the dataloader for validation
    dataset_valid = DatasetMusic(train=False, spectrogram=True)
    dataset_test = DatasetMusicTest(spectrogram=True)

    # Setup dataloader
    dataloader_test = DataLoader(dataset_test, batch_size=1)

    # Get the anchor waveforms
    anchors_waveforms = dataset_valid.get_anchors()
    anchors_embedding = get_anchor_embeddings(network, anchors_waveforms)

    distances = []

    for sample in tqdm(dataloader_test, desc="Validating", colour="red"):

        # Waveforms
        waveform = sample[0]
        filename = sample[1][0]

        # Generate prediction
        prediction = network(waveform)

        # Need to compare the prediction versus all anchors
        min_dist = 1000
        for key, val in anchors_embedding.items():

            # We are going to use Euclidean distance to figure out what to associate the embedding with
            dist = np.linalg.norm(prediction.detach().numpy() - val)
            if dist < min_dist:
                min_dist = dist

        # Check if dist is out of bounds
        distances.append((filename, min_dist))

    # Print out the mininum distances
    df = pd.DataFrame(distances, columns=["filename", "min_dist"])
    df = df.sort_values("min_dist").reset_index(drop=True)
    print(df)
    print("done!")


if __name__ == "__main__":

    plot_confusion_matrix()
    # infer_test_data()
