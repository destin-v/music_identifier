import os
from glob import glob

from midi2audio import FluidSynth
from tqdm import tqdm

from src.helper import print_midi


def convert_labeled_data_to_wav():
    """Converts the labeled data from MIDI format to WAV format."""

    fs = FluidSynth()

    # Pathing of files
    abs_dir = os.getcwd()
    source_dir = "Part One Data/Part1(PS1)"

    # Create iterator for musician folders
    parent_folders = os.listdir(source_dir)
    musician_folders = tqdm(parent_folders, colour="green")

    # Iterate over all musician folders
    for top_folder in musician_folders:
        musician_folders.set_description(f"Processing: {top_folder}")

        # Ignore irrelevant files
        if top_folder == ".DS_Store":
            continue  # this is a MacOSX generated file

        # Create iterator for MIDI files
        midi_files = tqdm(glob(source_dir + "/" + top_folder + "/*.mid"), colour="red")

        # Iterate over all files under musician folders
        for midi_file in midi_files:
            midi_files.set_description(f"Processing: {midi_file.split('/')[-1]}")
            input_file = abs_dir + "/" + midi_file
            output_file = os.path.splitext(input_file)[0] + ".wav"
            fs.midi_to_audio(input_file, output_file)


def convert_unlabled_data_to_wav():
    """Converts the unlabled data from MIDI format to WAV format."""

    fs = FluidSynth()

    # Pathing of files
    abs_dir = os.getcwd()
    source_dir = "Part One Data/Part1(PS2)"

    # Create iterator for MIDI files
    midi_files = tqdm(glob(source_dir + "/*.mid"), colour="green")

    # Iterate over all files under musician folders
    for midi_file in midi_files:
        midi_files.set_description(f"Processing: {midi_file.split('/')[-1]}")
        input_file = abs_dir + "/" + midi_file
        output_file = os.path.splitext(input_file)[0] + ".wav"
        fs.midi_to_audio(input_file, output_file)


# using the default sound font in 44100 Hz sample rate
if __name__ == "__main__":

    # First convert all MIDI files to WAV format
    convert_labeled_data_to_wav()
    convert_unlabled_data_to_wav()

    # View Track Metadata
    print_midi("Part One Data/Part1(PS2)/0.002716920481628_adj.mid")
