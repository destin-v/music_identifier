import os
import random
from glob import glob
from random import choice

import numpy as np
import pandas as pd
import torch
import torchaudio
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchaudio import transforms
from tqdm import tqdm


# Set random seed
random.seed(10)


def get_spectrogram(
    waveform: torch.Tensor,
    sample_rate: int,
    n_mels=64,
    n_fft=1024,
    hop_len=None,
    debug: bool = False,
) -> torch.TensorType:
    """This converts the audio file into a tensor with 2 channels with length and width which can be represented as an image.  This is easier to work with than the raw audio tensor format.

    Args:
        waveform (torch.Tensor): Audio data
        sample_rate (int): Sample rate (should be 44.1kHz)
        n_mels (int, optional): _description_. Defaults to 64.
        n_fft (int, optional): _description_. Defaults to 1024.
        hop_len (_type_, optional): _description_. Defaults to None.

    Returns:
        torch.TensorType: Tensor of shape 2x64x19654
    """
    # Cite: https://towardsdatascience.com/audio-deep-learning-made-simple-sound-classification-step-by-step-cebc936bbe5

    top_db = 80

    # spec has shape [channel, n_mels, time], where channel is mono, stereo etc
    spectrogram = transforms.MelSpectrogram(
        sample_rate, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels
    )(torch.Tensor(waveform))

    # Convert to decibels
    spectrogram = transforms.AmplitudeToDB(top_db=top_db)(spectrogram)

    check1 = spectrogram.shape[0] == 2
    check2 = spectrogram.shape[1] == 64
    check3 = spectrogram.shape[2] == 2584

    if (check1 and check2 and check3) is False:
        raise ValueError("Invalid sizing for Mel Spectogram!")

    if debug:
        plt.figure()
        plt.imshow(spectrogram.log2()[0, :, :].numpy(), cmap="viridis")
        plt.close()

    return spectrogram


class DatasetMusic(Dataset):
    def __init__(
        self,
        train: bool = True,
        spectrogram: bool = False,
    ) -> None:
        super().__init__()
        """Generate a dataframe that specifies the training and validation split of the data.
        """

        # Set path to source data
        self.source_dir = "Part One Data/Part1(PS1)"

        # Initialize variables
        self._df = pd.DataFrame
        self._df_path = "save/music_df.pkl"
        self._new_sample_rate = 44100
        self._train = train
        self.spectogram = spectrogram

        self.N_ANCHOR = 1
        self.N_POSITIVE = 1
        self.N_NEGATIVE = 1
        self.DEBUG = False

        if os.path.isfile(self._df_path):
            self._df = pd.read_pickle(self._df_path)
        else:
            self.generate_df()

    def generate_df(self):
        """Generate a dataframe that has the training and validation data."""

        data = []

        # Create iterator for musician folders
        parent_folders = os.listdir(self.source_dir)
        musician_folders = tqdm(parent_folders, colour="green")

        # Iterate over all musician folders
        for label in musician_folders:
            musician_folders.set_description(f"Creating Dataset: {label}")

            # Ignore irrelevant files
            if label == ".DS_Store":
                continue  # this is a MacOSX generated file

            # Create iterator for MIDI files
            wav_files_path = tqdm(
                glob(self.source_dir + "/" + label + "/*.wav"), colour="red"
            )

            # Iterate over all files under musician folders
            for ii, wav_file_path in enumerate(wav_files_path):

                # Create a flag for train/validation
                prob = random.random()
                if ii == 1:
                    anchor = True
                    flag_train = True
                    flag_valid = True
                elif prob <= 0.8:
                    anchor = False
                    flag_train = True
                    flag_valid = False
                else:
                    anchor = False
                    flag_train = False
                    flag_valid = True

                data.append((wav_file_path, label, anchor, flag_train, flag_valid))

        # Create a dataframe that contains all the wav files and their labels
        self._df = pd.DataFrame(
            data, columns=["wav_file_path", "label", "anchor", "train", "valid"]
        )

        self._df.to_pickle(self._df_path)

    def __getitem__(self, index: int) -> tuple:
        """Get an item from the dataset (i.e. data and label)

        Args:
            index (int): Unused at the moment.
            train (bool, optional): The training flag. Defaults to True.

        Returns:
            tuple: data, label
        """
        # Define musicians to choose from
        musicians = ["Bach", "Beethoven", "Brahms", "Schubert"]

        # Randomly select a musician
        anchor_musician = choice(musicians)

        # Filter on the selected musician
        filter_positive = self._df["label"] == anchor_musician
        filter_negative = self._df["label"] != anchor_musician
        filter_training = self._df["train"] == True
        filter_validate = self._df["valid"] == True
        filter_anchor = self._df["anchor"] == True

        # Refine based on training vs validation
        if self._train is True:
            filter_positive = filter_positive & filter_training
            filter_negative = filter_negative & filter_training
        if self._train is False:
            filter_positive = filter_positive & filter_validate
            filter_negative = filter_negative & filter_validate

        # Get positive and negative data entries
        filter_data_anc = self._df[filter_positive & filter_anchor]
        filter_data_pos = self._df[filter_positive & ~filter_anchor]
        filter_data_neg = self._df[filter_negative]

        # Extract the selected info
        df_anc = filter_data_anc.sample(n=self.N_ANCHOR).reset_index(drop=True)
        df_pos = filter_data_pos.sample(n=self.N_POSITIVE).reset_index(drop=True)
        df_neg = filter_data_neg.sample(n=self.N_NEGATIVE).reset_index(drop=True)

        anc_path_ = df_anc["wav_file_path"][0]
        pos_path_ = df_pos["wav_file_path"][0]
        neg_path_ = df_neg["wav_file_path"][0]

        anc_label = df_anc["label"][0]
        pos_label = df_pos["label"][0]
        neg_label = df_neg["label"][0]

        # Load the torchaudio waveform
        waveform_anc, sample_rate_anc = torchaudio.load(anc_path_)
        waveform_pos, sample_rate_pos = torchaudio.load(pos_path_)
        waveform_neg, sample_rate_neg = torchaudio.load(neg_path_)

        waveform_anc = self.resample(waveform_anc, sample_rate_anc, anc_path_)
        waveform_pos = self.resample(waveform_pos, sample_rate_pos, pos_path_)
        waveform_neg = self.resample(waveform_neg, sample_rate_neg, neg_path_)

        # Check shape of waveform
        if waveform_anc.shape != (2, 1323001):
            print("Invalid shape detected!")
        if waveform_pos.shape != (2, 1323001):
            print("Invalid shape detected!")
        if waveform_neg.shape != (2, 1323001):
            print("Invalid shape detected!")

        if self.spectogram:
            waveform_anc = get_spectrogram(waveform_anc, self._new_sample_rate)
            waveform_pos = get_spectrogram(waveform_pos, self._new_sample_rate)
            waveform_neg = get_spectrogram(waveform_neg, self._new_sample_rate)

        return waveform_anc, waveform_pos, waveform_neg, anc_label, pos_label, neg_label

    def __len__(self):
        if self._train:
            return sum(self._df["train"] == True)
        else:
            return sum(self._df["valid"] == True)

    def get_anchors(self) -> dict:
        """Retrieve all anchors from the dataset."""

        my_dict = {}

        # Extract the selected info
        df_anc = self._df[self._df["anchor"]]

        for ii, row in df_anc.iterrows():

            # Extract row information
            anc_path_ = row["wav_file_path"]
            anc_label = row["label"]

            # Load the torchaudio waveform
            waveform_anc, sample_rate_anc = torchaudio.load(anc_path_)

            # Resample to specified window
            waveform_anc = self.resample(waveform_anc, sample_rate_anc, anc_path_)

            # Convert to spectrogram
            if self.spectogram:
                waveform_anc = get_spectrogram(waveform_anc, self._new_sample_rate)

            my_dict[anc_label] = waveform_anc

        return my_dict

    def resample(self, waveform: torch.Tensor, sample_rate: int, path: str):

        # Conversion to 44.1kHz if not already
        if sample_rate != self._new_sample_rate:
            # Resample the data to match 44.1kHz sampling rate
            waveform = torchaudio.transforms.Resample(sample_rate, self._new_sample_rate)(
                waveform
            )
            print(f"Waveform Resampled: {path}")

        # Crop out 30 seconds of data randomly from the music file
        waveform = waveform.numpy()
        num_channels, num_frames = waveform.shape
        time_axis = np.arange(0, num_frames) / sample_rate

        # Set window for 30 second interval
        min_time_axis = 0
        max_time_axis = 30
        sel_time_axis = (time_axis >= min_time_axis) & (time_axis <= max_time_axis)
        waveform = waveform[:, sel_time_axis]

        # Adding debugging statement
        if self.DEBUG:
            torchaudio.save(
                "test_dataset_resample.wav",
                torch.tensor(waveform),
                sample_rate=self._new_sample_rate,
            )

        return waveform


class DatasetMusicTest(Dataset):
    def __init__(self, spectrogram: bool = False) -> None:
        super().__init__()
        """Generate a dataframe that is used to test the data."""

        # Set path to source data
        self.source_dir = "Part One Data/Part1(PS2)"

        # Initialize variables
        self._df = pd.DataFrame
        self._df_path = "save/test_music_df.pkl"
        self._new_sample_rate = 44100
        self.spectogram = spectrogram

        if os.path.isfile(self._df_path):
            self._df = pd.read_pickle(self._df_path)
        else:
            self.generate_df()

    def generate_df(self):

        data = []

        # Create iterator for MIDI files
        wav_files_path = tqdm(glob(self.source_dir + "/*.wav"), colour="red")

        # Iterate over all files under musician folders
        for ii, wav_file_path in enumerate(wav_files_path):
            data.append(wav_file_path)

        # Create a dataframe that contains all the wav files and their labels
        self._df = pd.DataFrame(data, columns=["wav_file_path"])

        self._df.to_pickle(self._df_path)

    def __getitem__(self, index: int) -> torch.Tensor:
        """Get an item from the dataset (i.e. data only)

        Args:
            index (int): Unused at the moment.

        Returns:
            tuple: data, label
        """

        # Get positive and negative data entries
        indexed_data = self._df.loc[index]

        # Extract the selected info
        path = indexed_data["wav_file_path"]
        name = str(path).split("/")[-1]

        # Load the torchaudio waveform
        waveform, sample_rate = torchaudio.load(path)
        waveform = self.resample(waveform, sample_rate, path)

        # Check shape of waveform
        if waveform.shape != (2, 1323001):
            print("Invalid shape detected!")

        if self.spectogram:
            waveform = get_spectrogram(waveform, self._new_sample_rate)

        return waveform, name

    def __len__(self):
        return len(self._df)

    def resample(self, waveform: torch.Tensor, sample_rate: int, path: str):

        # Conversion to 44.1kHz if not already
        if sample_rate != self._new_sample_rate:
            # Resample the data to match 44.1kHz sampling rate
            waveform = torchaudio.transforms.Resample(sample_rate, self._new_sample_rate)(
                waveform
            )
            print(f"Waveform Resampled: {path}")

        # Crop out 30 seconds of data randomly from the music file
        waveform = waveform.numpy()
        num_channels, num_frames = waveform.shape
        time_axis = np.arange(0, num_frames) / sample_rate

        # Set window for 30 second interval
        min_time_axis = 0
        max_time_axis = 30
        sel_time_axis = (time_axis >= min_time_axis) & (time_axis <= max_time_axis)
        waveform = waveform[:, sel_time_axis]

        return waveform


if __name__ == "__main__":

    # Setup training/validation dataset
    dataset_music_train = DatasetMusic(train=True)
    dataset_music_valid = DatasetMusic(train=False)

    # Setup training/validation dataset
    dataset_music_train.spectogram = True
    dataset_music_valid.spectogram = True

    # Convert to iterators
    data_train = iter(dataset_music_train)
    data_valid = iter(dataset_music_valid)

    # Test training iterator
    for ii in range(0, 10):
        data_train.__next__()

    # Test validation iterator
    for ii in range(0, 10):
        data_valid.__next__()

    # Create dataloaders for training/validation
    dataloader_valid = DataLoader(
        dataset_music_train,
        batch_size=4,
        shuffle=True,
    )

    dataloader_valid = DataLoader(
        dataset_music_valid,
        batch_size=4,
        shuffle=True,
    )
