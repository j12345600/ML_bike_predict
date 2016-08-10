import csv
import numpy as np

def csv2ndary(fname):
    with open(fname,newline="") as f:
        reader = csv.reader(f)
        return np.array(list(reader),dtype=float)
def csv_shape(fname):
    npary = csv2ndary(fname)
    return npary.shape
def get_total(fname):
    with open(fname,newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            return int(row[0])+int(row[1])
