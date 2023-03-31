# @markdown In this tutorial, we will use a speech data from [VOiCES dataset](https://iqtlabs.github.io/voices/), which is licensed under Creative Commos BY 4.0.
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from mido import MidiFile

from src.data import DatasetMusic


[width, height] = matplotlib.rcParams["figure.figsize"]
if width < 10:
    matplotlib.rcParams["figure.figsize"] = [width * 2.5, height]


def print_stats(waveform: torch.Tensor, sample_rate: int = None, src: str = None):
    """Prints out general stats.

    Args:
        waveform (tensor): Audio waveform
        sample_rate (int, optional): Sample rate. Defaults to None.
        src (path, optional): Path of file. Defaults to None.
    """
    if src:
        print("-" * 10)
        print("Source:", src)
        print("-" * 10)
    if sample_rate:
        print("Sample Rate:", sample_rate)
    print("Shape:", tuple(waveform.shape))
    print("Dtype:", waveform.dtype)
    print(f" - Max:     {waveform.max().item():6.3f}")
    print(f" - Min:     {waveform.min().item():6.3f}")
    print(f" - Mean:    {waveform.mean().item():6.3f}")
    print(f" - Std Dev: {waveform.std().item():6.3f}")
    print()
    print(waveform)
    print()


def print_midi(path: str):
    """Prints out general MIDI information.

    Args:
        path (_type_): _description_
    """
    mid = MidiFile(path, clip=True)
    print(mid)


def plot_waveform(
    waveform: torch.Tensor,
    sample_rate: int,
    title: str = "Waveform",
    xlim: int = None,
    ylim: int = None,
):
    """_summary_

    Args:
        waveform (torch.Tensor): Audio waveform.
        sample_rate (int): Sample rate.
        title (str, optional): Title. Defaults to "Waveform".
        xlim (int, optional): Limits. Defaults to None.
        ylim (int, optional): Limits. Defaults to None.
    """
    waveform = np.array(waveform)

    num_channels, num_frames = waveform.shape
    time_axis = np.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].plot(time_axis, waveform[c], linewidth=1)
        axes[c].grid(True)
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c+1}")
        if xlim:
            axes[c].set_xlim(xlim)
        if ylim:
            axes[c].set_ylim(ylim)
    figure.suptitle(title)
    plt.show(block=False)


def plot_specgram(
    waveform: torch.Tensor,
    sample_rate: int,
    title: str = "Spectrogram",
    xlim: int = None,
):
    """Plots the spectogram as 2 channels.

    Args:
        waveform (torch.Tensor): Audio waveform.
        sample_rate (int): Sample rate.
        title (str, optional): Title. Defaults to "Spectrogram".
        xlim (int, optional): Limit. Defaults to None.
    """
    waveform = np.array(waveform)

    num_channels, num_frames = waveform.shape

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        sliced_waveform = waveform[c]
        axes[c].specgram(sliced_waveform, Fs=sample_rate)
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c+1}")
        if xlim:
            axes[c].set_xlim(xlim)

    figure.suptitle(title)
    plt.show(block=False)


def plot_music():
    """Plotting function for debugging and checking purposes."""

    # Template from: https://pytorch.org/tutorials/beginner/audio_preprocessing_tutorial.html#loading-audio-data-into-tensor

    dataset = iter(DatasetMusic())

    sample_rate = 44100

    waveform_anc, waveform_pos, waveform_neg, _, _, _ = next(dataset)
    print_stats(waveform_anc, sample_rate=sample_rate)
    plot_waveform(waveform_anc, sample_rate)
    plot_specgram(waveform_anc, sample_rate)


if __name__ == "__main__":
    plot_music()
    print("done")
