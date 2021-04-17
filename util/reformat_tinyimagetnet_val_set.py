import shutil
import os
from tqdm import tqdm


def mkdir(path: str):
    if not os.path.exists(path):
        os.mkdir(path)

VAL_DIRECTORY = "D:/VUT/KNN/project/data/tiny-imagenet-200/val"
OUTPUT_DIRECTORY = "D:/VUT/KNN/project/data/tiny-imagenet-200/test"
with open(os.path.join(VAL_DIRECTORY, "val_annotations.txt"), "r") as f:
    annotations = list(map(lambda x: x.split("\t"), filter(None, f.read().split("\n"))))

mkdir(OUTPUT_DIRECTORY)
for label in set(map(lambda x: x[1], annotations)):
    mkdir(os.path.join(OUTPUT_DIRECTORY, label))

for filename, label, *_ in tqdm(annotations):
    shutil.copyfile(
        src=os.path.join(VAL_DIRECTORY, "images", filename),
        dst=os.path.join(OUTPUT_DIRECTORY, label, filename)
    )
