import tarfile
import pathlib
import random
import shutil
import os
import glob
from tqdm import tqdm, trange
from modfire.utils import getRichProgress
from PIL import Image

from torchvision.datasets import ImageNet



try:
    os.remove("temp.tar.gz")
except FileNotFoundError:
    pass
try:
    for chunk in glob.glob("*.tar.gz.*"):
        os.remove(chunk)
except FileNotFoundError:
    pass


# Prepare the ImageNet folder by https://github.com/PatrickHua/EasyImageNet

LABELS = list(range(1000))
for _ in range(10):
    random.shuffle(LABELS)
randomClass = sorted(LABELS[:100])

zeroBaseMapping = {
    x: i for i, x in enumerate(randomClass)
}
print(zeroBaseMapping)

trainSet = ImageNet("./imagenet")

databaseImgs = { x[0]: zeroBaseMapping[x[1]] for x in filter(lambda x: x[1] in randomClass, trainSet.imgs)}

valSet = ImageNet("./imagenet", "val")

queryImgs = { x[0]: zeroBaseMapping[x[1]] for x in filter(lambda x: x[1] in randomClass, valSet.imgs)}

# a tar with only images in root
with tarfile.open("temp.tar.gz", mode="w:gz") as new:
    for img in tqdm(databaseImgs.items(), desc="Adding database to temp.tar.gz", leave=True):
        imgPath, imgLabel = img
        # remove corrupted image
        try:
            Image.open(imgPath).verify()
        except:
            print(f"{imgPath} corrupted.")
            if img in databaseImgs:
                databaseImgs.pop(img)
                print("popped from databaseImgs")
            # if img in unusedFiles:
            #     unusedFiles.pop(img)
            #     print("popped from unusedFiles")
            continue
        name = imgPath[len("./imagenet/"):]
        new.add(imgPath, name, recursive=True)

    for img in tqdm(queryImgs.items(), desc="Adding query to temp.tar.gz", leave=True):
        imgPath, imgLabel = img
        # remove corrupted image
        try:
            Image.open(imgPath).verify()
        except:
            print(f"{imgPath} corrupted.")
            if img in queryImgs:
                queryImgs.pop(img)
                print("popped from queryImgs")
            # if img in unusedFiles:
            #     unusedFiles.pop(img)
            #     print("popped from unusedFiles")
            continue
        name = imgPath[len("./imagenet/"):]
        new.add(imgPath, name, recursive=True)


trainList = []
buckets = [0 for _ in range(100)]
for img in databaseImgs.items():
    imgPath, label = img
    if buckets[label] > 99:
        continue
    buckets[label] += 1
    trainList.append(imgPath[len("./imagenet/"):] + " " + str(label) + os.linesep)

assert len(trainList) == 10000, len(trainList)
with open("train.txt", "w") as fp:
        fp.writelines(trainList)

databaseImgs = [key[len("./imagenet/"):] + " " + str(value) + os.linesep for key, value in databaseImgs.items()]
with open("database.txt", "w") as fp:
    fp.writelines(databaseImgs)

queryImgs = [key[len("./imagenet/"):] + " " + str(value) + os.linesep for key, value in queryImgs.items()]
with open("query.txt", "w") as fp:
    fp.writelines(queryImgs[-5000:])
# with open("unused.txt", "w") as fp:
#     fp.writelines(unusedImageList)


def chunks(fp, size):
    while content := fp.read(size):
        yield content


# chunk tar.gz into 1.99GiB files
with open("temp.tar.gz", "rb") as fp:
    for i, chunk in enumerate(chunks(fp, 1024*1024*1000*2)):
        with open(f"ImageNet.tar.gz.{i}", "wb") as fp:
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
# IMPORTANT: USED IN modfire.dataset.easy.nuswide
print(f"All check passed. Global hash: {originalHash}")

with open("labelMap.txt", "w") as fp:
    fp.writelines(str(k) + os.linesep for k, v in sorted(zeroBaseMapping.items(), key=lambda x: x[1]))
