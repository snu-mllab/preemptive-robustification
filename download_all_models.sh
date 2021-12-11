#!/bin/bash

for subdir in ./checkpoints/*/*/*/; do
    echo $subdir
    ( cd "$subdir"; bash ./download.sh ) || break
    echo
done