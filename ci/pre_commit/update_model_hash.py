import os
import sys
import requests
import re


with open(sys.argv[1]) as fp:
    lines = fp.readlines()


flag = False
start = 0
end = 0


for i, line in enumerate(lines):
    if "MODELS_HASH" in line:
        flag = True
        start = i
        continue
    if flag:
        if line.startswith("}"):
            end = i
            break

MODELS_HASH = dict()

response = requests.get("https://api.github.com/repos/VL-Group/modfire/releases/tags/generic", headers={"Accept":"application/vnd.github.v3+json"}).json()
assets = response["assets"]


for asset in assets:
    name: str = asset["name"]
    if not name.endswith("modfire"):
        continue

    regex = re.compile(r"^qp_[0-9]*_(mse|msssim)_[0-9a-fA-F]{8,}\.modfire$")
    if not regex.match(name):
        raise ValueError(f"Naming convention broken with `{name}`.")

    stem = name.split(".")[0]
    component = stem.split("_")
    qp = component[1]
    target = component[2]
    hashStr = component[-1]
    print(qp, target, hashStr)
    MODELS_HASH[f"qp_{qp}_{target}"] = hashStr

MODELS_HASH = """MODELS_HASH = {
%s
}
""" % (os.linesep.join(f"    \"{key}\": \"{value}\"" for key, value in MODELS_HASH.items()))

result = lines[:start] + [MODELS_HASH] + lines[(end+1):]

with open(sys.argv[1], "w") as fp:
    fp.writelines(result)
