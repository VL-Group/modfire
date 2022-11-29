import os
import sys


__ENTRY_POINTS__ = {
    "modfire": "modfire.cli:entryPoint",
    "modfire-train": "modfire.train.cli:entryPoint",
    "modfire-dataset": "modfire.datasets.cli:entryPoint",
    "modfire-validate": "modfire.validate.cli:entryPoint"
}


def writeYAML(lines):
    result = list()
    for line in lines:
        if "${{ ENTRY_POINTS }}" in line:
            indent = line.rstrip().replace("${{ ENTRY_POINTS }}", "").replace('\r', '').replace('\n', '')
            replaced = [f"{indent}- {key} = {value}{os.linesep}" for key, value in __ENTRY_POINTS__.items()]
            result.extend(replaced)
        else:
            result.append(line)
    return result


if __name__ == "__main__":
    with open(sys.argv[1]) as fp:
        result = writeYAML(fp.readlines())
    with open(sys.argv[1], "w") as fp:
        fp.writelines(result)
