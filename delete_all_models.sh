#!/bin/bash

for subdir in ./checkpoints/*/*/*/; do
    echo "Path: $subdir"
    echo "Deleting a checkpoint..."
    ( cd "$subdir"; rm ckpt.pt ) || break
    echo
done
