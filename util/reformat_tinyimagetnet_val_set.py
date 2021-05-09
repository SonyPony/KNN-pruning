# Authors: Son Hai Nguyen, Miroslav Karpíšek
# Logins: xnguye16, xkarpi05
# Project: Neural network pruning
# Course: Convolutional Neural Networks
# Year: 2021


import shutil
import os
from tqdm import tqdm
import argparse


def mkdir(path: str):
    if not os.path.exists(path):
        os.mkdir(path)

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, required=True, help="Path to the dataset root directory.")
args = parser.parse_args()

val_directory = os.path.join(args.dataset, "val")
output_directory = os.path.join(args.dataset, "test")
with open(os.path.join(val_directory, "val_annotations.txt"), "r") as f:
    annotations = list(map(lambda x: x.split("\t"), filter(None, f.read().split("\n"))))

mkdir(output_directory)
for label in set(map(lambda x: x[1], annotations)):
    mkdir(os.path.join(output_directory, label))

for filename, label, *_ in tqdm(annotations):
    shutil.copyfile(
        src=os.path.join(val_directory, "images", filename),
        dst=os.path.join(output_directory, label, filename)
    )
