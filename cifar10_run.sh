#!/bin/bash

for i in {10..14}
do
   echo "Run $i model : $1"
   python main.py --model=$1 --random_seed=$i
done

