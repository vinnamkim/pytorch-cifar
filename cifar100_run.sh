#!/bin/bash

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"

for i in {10..14}
do
   echo "Run $i model : $1"
   python $SCRIPT_DIR/main.py --dataset=CIFAR100 --model=$1 --random-seed=$i --epoch=90 --num-workers=$2 --batch-size=$3
done
