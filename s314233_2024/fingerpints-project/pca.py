import numpy
import pandas as pd
import os

import matplotlib.pyplot as plt

def vcol(x):
    return x.reshape((x.size, 1))

def vrow(x):
    return x.reshape((1, x.size))

def compute_mu_C(D):
    mu = vcol(D.mean(1))
    C = ((D-mu) @ (D-mu).T) / float(D.shape[1])
    return mu, C

def compute_pca(D, m):

    mu, C = compute_mu_C(D)
    U, s, Vh = numpy.linalg.svd(C)
    P = U[:, 0:m]
    return P

def apply_pca(P, D):
    return P.T @ D
    
def load(pathname, visualization=0):
    df = pd.read_csv(pathname, header=None)
    if visualization:
        print(df.head())
    attribute = numpy.array(df.iloc[:, 0 : len(df.columns) - 1]).T
    label = numpy.array(df.iloc[:, -1])
    return attribute, label
if __name__ == '__main__':

    path = os.path.abspath("Train.txt")
    D, L = load(path)
    mu, C = compute_mu_C(D)
    print(mu)
    print(C)
    P = compute_pca(D, m = 5)
    print(P)
