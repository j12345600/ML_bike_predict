import utilities as utl
import numpy as np
import random
Path_test="./dataset/61/train61.csv"

def local_split(train_index):
   random.seed(0)
   train_index = set(train_index)
   all_index = sorted(train_index)
   num_test = int(len(all_index) / 3)
   random.shuffle(all_index)
   train_set = set(all_index[num_test:])
   test_set = set(all_index[:num_test])
   return train_set, test_set


def split_csv(src_csv, split_to_train, train_csv, test_csv):
   ftrain = open(train_csv, "w")
   ftest = open(test_csv, "w")
   cnt = 0
   for l in open(src_csv):
       if split_to_train[cnt]:
           ftrain.write(l)
       else:
           ftest.write(l)
       cnt = cnt + 1
   ftrain.close()
   ftest.close()
