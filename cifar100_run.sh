#!/bin/bash

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"

for i in {10..14}
do
   echo "Run $i model : $1"
   python $SCRIPT_DIR/cifar_100.py --model=$1 --random_seed=$i --epoch=90
done

