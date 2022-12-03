import tarfile
import pathlib
import random
import shutil
import os
import glob
from tqdm.rich import tqdm, trange
from modfire.utils import getRichProgress

# Download from https://lms.comp.nus.edu.sg/wp-content/uploads/2019/research/nuswide/NUS-WIDE.html
with open("Imagelist.txt", "r") as fp:
    allImages = [x.strip().split("\\")[-1] for x in fp.readlines()]

with open("AllTags81.txt", "r") as fp:
    allTags = fp.readlines()

# remove entries that have all-zero tag.
# As for test in 2022, there are 136,207 images have none of labels.
# Then, database only has 128,441 images.
allFiles = dict(zip(allImages, allTags))
unusedFiles = dict(filter(lambda i: not any(map(lambda x: x > 0, map(int, i[1].strip().split()))), zip(allImages, allTags)))


filtered = dict()
filtered = allFiles

try:
    os.remove("temp.tar.gz")
except FileNotFoundError:
    pass
shutil.rmtree("temp", ignore_errors=True)
try:
    for chunk in glob.glob("*.tar.gz.*"):
        os.remove(chunk)
except FileNotFoundError:
    pass

# Download from https://github.com/swuxyj/DeepHash-pytorch
with tarfile.open("NUS-WIDE.tar.gz", mode="r:gz") as original:
    print("extracting...")
    original.extractall("temp")

# a tar with only images in root
with tarfile.open("temp.tar.gz", mode="w:gz") as new:
    allImages = glob.glob("temp/**/*.jpg", recursive=True)
    i = 0
    for img in tqdm(allImages, desc="Adding to temp.tar.gz", leave=False):
        path = pathlib.Path(img)
        name = path.name
        filtered[name] = allFiles[name]
        new.add(img, name, recursive=False)
        i += 1

for key in unusedFiles.keys():
    filtered.pop(key)

imageList = [f"{key} {value}" for key, value in filtered.items()]
unusedImageList = [f"{key} {value}" for key, value in unusedFiles.items()]

for i in trange(100, desc="Shuffling", leave=False):
    random.shuffle(imageList)

with open("train.txt", "w") as fp:
    fp.writelines(imageList[:10000])

with open("database.txt", "w") as fp:
    fp.writelines(imageList[:-5000])

with open("query.txt", "w") as fp:
    fp.writelines(imageList[-5000:])
with open("unused.txt", "w") as fp:
    fp.writelines(unusedImageList)


def chunks(fp, size):
    while content := fp.read(size):
        yield content


# chunk tar.gz into 1.99GiB files
with open("temp.tar.gz", "rb") as fp:
    for i, chunk in enumerate(chunks(fp, 1024*1024*1000*2)):
        with open(f"NUS-WIDE.tar.gz.{i}", "wb") as fp:
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
