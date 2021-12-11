#!/bin/bash

for subdir in ./checkpoints/*/*/*/; do
    echo "Path: $subdir"
    echo "Downloading a checkpoint..."
    ( cd "$subdir"; bash ./download.sh ) || break
    echo
done
