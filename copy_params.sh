#!/bin/bash
cp data/W1_$2.txt data/W1_$1.txt
cp data/W2_$2.txt data/W2_$1.txt
cp data/b1_$2.txt data/b1_$1.txt
cp data/b2_$2.txt data/b2_$1.txt

if test -f data/x_train_$2.txt; then
  cp data/x_train_$2.txt data/x_train_$1.txt
  cp data/x_test_$2.txt data/x_test_$1.txt
  cp data/y_train_$2.txt data/y_train_$1.txt
  cp data/y_test_$2.txt data/y_test_$1.txt
fi
