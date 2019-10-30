#!/bin/bash

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"

for i in {10..14}
do
   save = False
   if [${i} -eq 10];then
      save = True
   fi
   echo "Run $i model : $1 save : ${save}"
   python $SCRIPT_DIR/main.py --dataset=CIFAR10 --model=$1 --random-seed=$i --epoch=90 --num-workers=$2 --batch-size=$3
done

