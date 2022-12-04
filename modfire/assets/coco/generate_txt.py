import tarfile
import pathlib
import random
import shutil
import os
import glob
from tqdm.rich import tqdm, trange
from modfire.utils import getRichProgress
import zipfile
import json
import torch
from PIL import Image

torch.hub.download_url_to_file("http://images.cocodataset.org/zips/train2017.zip", "./train2017.zip")
torch.hub.download_url_to_file("http://images.cocodataset.org/zips/val2017.zip", "./val2017.zip")
torch.hub.download_url_to_file("http://images.cocodataset.org/zips/annotations_trainval2017.zip", "./")



with zipfile.ZipFile("train2017.zip") as zippedFile:
    zippedFile.extractall("./")
with zipfile.ZipFile("val2017.zip") as zippedFile:
    zippedFile.extractall("./")
with zipfile.ZipFile("annotations_trainval2017.zip") as zippedFile:
    zippedFile.extractall("./")

with open("annotations/instances_train2017.json") as fp:
    train = json.load(fp)
with open("annotations/instances_val2017.json") as fp:
    val = json.load(fp)

categories = {x['id']: x['name'] for x in train["categories"]}
valCategories = {x['id']: x['name'] for x in val["categories"]}

assert all(v == valCategories[k] for k, v in categories.items())

idToIdxMap = {
    k: i for i, k in enumerate(sorted(categories))
}
sortedLabels = [categories[x] for x in sorted(categories)]

imageWithIds = {
    x['id']: os.path.join("train2017", x['file_name']) for x in train["images"]
}
imageWithIds.update({
    x['id']: os.path.join("val2017", x['file_name']) for x in val["images"]
})

annotations = dict()

for ann in train["annotations"] + val["annotations"]:
    if ann["image_id"] not in annotations:
        annotations[ann["image_id"]] = list()
    annotations[ann["image_id"]].append(ann["category_id"])

imageLabels = { key: [0] * len(categories) for key in annotations.keys() }

for key, value in annotations.items():
    for v in value:
        imageLabels[key][idToIdxMap[v]] = 1

# Iterate over train and image folder to remove corrupted files
allFiles = dict(filter(lambda k: any(map(lambda x: x > 0, k[1])), imageLabels.items()))
unusedFiles = dict(filter(lambda k: not any(map(lambda x: x > 0, k[1])), imageLabels.items()))

print("#unused: ", len(unusedFiles))

filtered = {}

try:
    os.remove("temp.tar.gz")
except FileNotFoundError:
    pass
try:
    for chunk in glob.glob("*.tar.gz.*"):
        os.remove(chunk)
except FileNotFoundError:
    pass

# a tar with only images in root
with tarfile.open("temp.tar.gz", mode="w:gz") as new:
    for imageId, imgLabel in tqdm(list(allFiles.items()), desc="Adding to temp.tar.gz", leave=False):
        imgPath = imageWithIds[imageId]
        if not any(map(lambda x: x > 0, imgLabel)):
            print(f"{imgPath} has all zero label")
            continue
        if not os.path.exists(imgPath):
            print(f"{imgPath} is missing.")
            continue
        # remove corrupted image
        try:
            Image.open(imgPath).verify()
        except:
            print(f"{imgPath} corrupted.")
            continue
        path = pathlib.Path(imgPath)
        name = path.name
        filtered[name] = imgLabel
        new.add(imgPath, name, recursive=False)


imageList = [str(key) + " " + " ".join(map(str, value)) + os.linesep for key, value in filtered.items()]

for i in trange(100, desc="Shuffling", leave=False):
    random.shuffle(imageList)

with open("train.txt", "w") as fp:
    fp.writelines(imageList[:10000])

with open("database.txt", "w") as fp:
    fp.writelines(imageList[:-5000])

with open("query.txt", "w") as fp:
    fp.writelines(imageList[-5000:])


def chunks(fp, size):
    while content := fp.read(size):
        yield content


# chunk tar.gz into 1.99GiB files
with open("temp.tar.gz", "rb") as fp:
    for i, chunk in enumerate(chunks(fp, 1024*1024*1000*2)):
        with open(f"COCO.tar.gz.{i}", "wb") as fp:
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
shutil.rmtree("temp", ignore_errors=True)
# IMPORTANT: USED IN modfire.dataset.easy.nuswide
print(f"All check passed. Global hash: {originalHash}")
