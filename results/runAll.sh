#!/bin/bash

rm -rf saved

CONFIGS="../configs/**/**/**/**"

for i in $CONFIGS/*.yaml; do
    modfire train $i
done

SAVED="saved/**/**/**/**/latest/best.ckpt"
for i in $SAVED; do
    dir1=$(echo $i | cut -d "/" -f 2)
    dir2=$(echo $i | cut -d "/" -f 3)
    dir3=$(echo $i | cut -d "/" -f 4)
    testYaml="$dir1/$dir2/$dir3/test.yaml"
    modfire validate $i $testYaml
done
