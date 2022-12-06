
import os
import numpy as np
import torch
import zipfile
import tarfile
import pathlib
import random
import shutil
import os
import glob
from tqdm import tqdm, trange
from modfire.utils import getRichProgress
from PIL import Image


# https://github.com/swuxyj/DeepHash-pytorch/blob/master/data/mirflickr/code.py
# Download http://press.liacs.nl/mirflickr/mirflickr25k.v3b/mirflickr25k.zip
# and http://press.liacs.nl/mirflickr/mirflickr25k.v3b/mirflickr25k_annotations_v080.zip
torch.hub.download_url_to_file("http://press.liacs.nl/mirflickr/mirflickr25k.v3b/mirflickr25k.zip", "./mirflickr25k.zip")
torch.hub.download_url_to_file("http://press.liacs.nl/mirflickr/mirflickr25k.v3b/mirflickr25k_annotations_v080.zip", "./mirflickr25k_annotations_v080.zip")


with zipfile.ZipFile("mirflickr25k.zip") as zippedFile:
    zippedFile.extractall("./") # ./mirflickr
with zipfile.ZipFile("mirflickr25k_annotations_v080.zip") as zippedFile:
    zippedFile.extractall("./mirflickr25k_annotations_v080") # ./mirflickr25k_annotations_v080

label_dir_name = "mirflickr25k_annotations_v080"

allLabelsTxt = sorted(list(filter(lambda x: ((not "_r1" in x) and (not "README" in x)), glob.glob(os.path.join(label_dir_name, "*.txt")))))

allLabels = list()
all_label_data = np.zeros((25000, 38), dtype=np.int8)
for i, label_file in enumerate(allLabelsTxt):
    allLabels.append(pathlib.Path(label_file).stem)
    with open(label_file, "r") as f:
        for line in f.readlines():
            all_label_data[int(line.strip()) - 1][i] = 1

print(allLabels)

allFilesTxt = list()

# a tar with only images in root
with tarfile.open("temp.tar.gz", mode="w:gz") as new:
    for i in trange(len(all_label_data), desc="Adding images to temp.tar.gz", leave=True):
        imgPath = os.path.join("mirflickr", f"im{i+1}.jpg")
        # remove corrupted image
        try:
            Image.open(imgPath).verify()
        except:
            continue
        allFilesTxt.append(f"im{i + 1}.jpg" + " " + " ".join(map(str, all_label_data[i].tolist())) + os.linesep)
        name = f"im{i+1}.jpg"
        new.add(imgPath, name, recursive=False)


train_num = 4000
test_num = 1000
for _ in range(10):
    random.shuffle(allFilesTxt)

train_data_index = allFilesTxt[:train_num]
test_data_index = allFilesTxt[train_num:train_num + test_num]
database_data_index = allFilesTxt[train_num + test_num:]

with open("database.txt", "w") as f:
    f.writelines(database_data_index)
with open("train.txt", "w") as f:
    f.writelines(train_data_index)
with open("query.txt", "w") as f:
    f.writelines(test_data_index)

def chunks(fp, size):
    while content := fp.read(size):
        yield content


# chunk tar.gz into 1.99GiB files
with open("temp.tar.gz", "rb") as fp:
    for i, chunk in enumerate(chunks(fp, 1024*1024*1000*2)):
        with open(f"mirflickr25k.tar.gz.{i}", "wb") as fp:
            fp.write(chunk)

# generate hash
from modfire.utils import hashOfFile, hashOfStream, concatOfFiles
chunkedFiles = glob.glob("*.tar.gz.*")
with getRichProgress() as p:
    for chunk in chunkedFiles:
        hashValue = hashOfFile(chunk, p)
        newName = chunk.split(".")
        newName = ".".join([newName[0] + f"_{hashValue[:8]}"] + newName[1:])
        shutil.move(chunk, newName)

    # check the hash of concated file equals to the original file
    originalHash = hashOfFile("temp.tar.gz", p)
    chunkedFiles = glob.glob("*.tar.gz.*")

with concatOfFiles(sorted(chunkedFiles, key=lambda x: int(x.split(".")[-1]))) as stream:

    print("Calculating combined hash...")
    combinedHash = hashOfStream(stream)
if originalHash != combinedHash:
    raise RuntimeError(f"Combine failed. original: {originalHash}. combined: {combinedHash}.")

os.remove("temp.tar.gz")
shutil.rmtree("mirflickr", ignore_errors=True)
shutil.rmtree("mirflickr25k_annotations_v080", ignore_errors=True)
# IMPORTANT: USED IN modfire.dataset.easy.nuswide
print(f"All check passed. Global hash: {originalHash}")

print(allLabels)
