#!/bin/bash

for subdir in ./checkpoints/*/*/*/; do
    echo "Path: $subdir"
    echo "Deleting a checkpoint..."
    ( cd "$subdir"; find . ! -name 'download.sh' -type f -exec rm -f {} + ) || break
    echo
done
