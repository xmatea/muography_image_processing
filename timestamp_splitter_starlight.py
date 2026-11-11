import os
import re
import h5py
import time
from astropy.io import fits

def read_metadata(nexus_files_path: str):
    metadata = {}
    for filename in os.listdir(nexus_files_path):
        if not filename.endswith(".nxs"):
            continue

        nexus_file_path = nexus_files_path + filename
        file = h5py.File(nexus_file_path, "r")

        data = file["raw_data_1"]
        start_time = data["start_time"][()].decode("utf-8")
        end_time = data["end_time"][()].decode("utf-8")
        comment = data["title"][()].decode("utf-8")
        muons = data[""]

        match = re.search(r"Momentum=(\d+.?\d+)", comment)
        if match is None:
            continue

        momentum = match.group(1)

        start_time_parsed = time.strptime(start_time, "%Y-%m-%d %H:%M:%S")
        end_time_parsed = time.strptime(end_time, "%Y-%m-%d %H:%M:%S")

        metadata[filename] = (start_time_parsed, end_time_parsed, comment, momentum)

        print(f"{filename} - {comment}: {start_time}, {end_time}")

    return metadata

def load_comment_nxs(input_file: h5py.File) -> tuple[list[str], int]:
    """
    Loads data from the appropriate comment subfolders in run file

    Args:
        input_file: HDF run file with comment metadata

    Returns:
        Returns a list containing [title, num prompt events, num delayed events, start time, end time, IBEX time cut for prompt, IBEX time cut for delayed] and an integer success flag.
        If no data was found, the list will be equal to ``[" ", " ", " ", " "]``.

    """
    try:
        encoding = "utf-8"

        title = input_file["title"][()].decode(encoding)
        title += ": " + input_file["notes"][()].decode(encoding)
        start_time = input_file["start_time"][()].decode(encoding)
        end_time = input_file["end_time"][()].decode(encoding)
        num_prompt_events = 0
        num_delayed_events = 0
        detector_channels = [1, 2, 3, 4]

        for i in detector_channels:
            num_prompt_events += input_file[f"detector_{i}_energyA/num_events"][()]
            num_delayed_events += input_file[f"detector_{i}_energyB/num_events"][()]
            try:  # Failsafe for now as not sure if all four channels will always record time
                key_Amin = f"detector_{i}_energyA/event_time_min"
                key_Amax = f"detector_{i}_energyA/event_time_max"
                prompt_time = (
                    f"{input_file[key_Amin][()]} - {input_file[key_Amax][()]} ns"
                )
                key_Bmin = f"detector_{i}_energyB/event_time_min"
                key_Bmax = f"detector_{i}_energyB/event_time_max"
                delayed_time = (
                    f"{input_file[key_Bmin][()]} - {input_file[key_Bmax][()]} ns"
                )
            except KeyError:
                pass

        comment_flag = 0
        rtn_str = [
            title,
            str(num_prompt_events),
            str(num_delayed_events),
            start_time,
            end_time,
            prompt_time,
            delayed_time,
        ]

    except IOError:
        rtn_str = [" ", " ", " ", " ", " ", " ", " "]
        comment_flag = 1
    return rtn_str, comment_flag

def sort_by_momentum(metadata: dict, path: str):

    for filename in os.listdir(path):
        if not filename.endswith(".fit"):
            continue

        file_path = path + filename

        hdul = fits.open(file_path)
        datestr = hdul[0].header['DATE-OBS']
        timestr = hdul[0].header['TIME-OBS']

        fullstr = datestr + " " + timestr

        creation_time = time.strptime(fullstr, "%a %b %d %Y %Z %H:%M:%S")
        #print(creation_time)

def load_counts(data_folder):
    counts = {}

    for filename in os.listdir(data_folder):
        if not filename.endswith(".nxs"):
            continue

        file_path = data_folder + filename
        file = h5py.File(file_path, "r")
        data = file["raw_data_1"]

        comment, flags = load_comment_nxs(data)

        match = re.search(r"Momentum=(\d+.?\d+)", comment[0])
        if match is None:
            continue

        momentum = float(match.group(1))
        delayed_events = int(comment[2])

        counts[momentum] = delayed_events

    return counts

if __name__ == "__main__":
    #starlight_files_path = "C:\\Users\\sec90771\\OneDrive - University College London\\Experiment data\\June26\\9V_Toshiba_images\\Bkg\\"
    nexus_files_path = "C:\\Users\\sec90771\\OneDrive - University College London\\Experiment data\\June26\\nexus\\AAAA\\"

    #metadata = read_metadata(nexus_files_path)



    for filename in os.listdir(nexus_files_path):
        file_path = nexus_files_path + filename

        if not filename.endswith(".nxs"):
            continue

        #print(f"{filename} - {file_path}")
        file = h5py.File(file_path, "r")
        data = file["raw_data_1"]

        comment, flags = load_comment_nxs(data)
        print(comment)
    #sort_by_momentum(metadata, starlight_files_path)
