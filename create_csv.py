#!/usr/bin/env python3
import os
import pandas as pd

CLASSES_MAP = {
    "rock": 0,
    "paper": 1,
    "scissor": 2,
    "none": 3
}

def mapper(val):
    return CLASSES_MAP[val]

if __name__ == "__main__":
    IMG_SAVE_PATH = "rps_images"

    dataset = []
    for directory in os.listdir(IMG_SAVE_PATH):
        # Ignore hidden files
        if directory.startswith("."):
            continue
        # Ignore sub folders
        if os.path.isdir(directory):
            continue

        label = mapper(directory.split("_")[0])
        dataset.append([directory, label])

    dataset = pd.DataFrame(dataset, columns=["Img", "Label"])
    dataset.to_csv("rps.csv", index=False)
