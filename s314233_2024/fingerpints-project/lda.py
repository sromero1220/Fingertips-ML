import numpy
import os
import pandas as pd
import matplotlib.pyplot as plt

import scipy.linalg

def vcol(x): # Same as in pca 
    return x.reshape((x.size, 1))

def vrow(x): # Same as in pca 
    return x.reshape((1, x.size))

def compute_mu_C(D): # Same as in pca 
    mu = vcol(D.mean(1))
    C = ((D-mu) @ (D-mu).T) / float(D.shape[1])
    return mu, C

def compute_Sb_Sw(D, L):
    Sb = 0
    Sw = 0
    muGlobal = vcol(D.mean(1))
    for i in numpy.unique(L):
        DCls = D[:, L == i]
        mu = vcol(DCls.mean(1))
        Sb += (mu - muGlobal) @ (mu - muGlobal).T * DCls.shape[1]
        Sw += (DCls - mu) @ (DCls - mu).T
    return Sb / D.shape[1], Sw / D.shape[1]

def compute_lda_geig(D, L, m):
    
    Sb, Sw = compute_Sb_Sw(D, L)
    s, U = scipy.linalg.eigh(Sb, Sw)
    return U[:, ::-1][:, 0:m]

def compute_lda_JointDiag(D, L, m):

    Sb, Sw = compute_Sb_Sw(D, L)

    U, s, _ = numpy.linalg.svd(Sw)
    P = numpy.dot(U * vrow(1.0/(s**0.5)), U.T)

    Sb2 = numpy.dot(P, numpy.dot(Sb, P.T))
    U2, s2, _ = numpy.linalg.svd(Sb2)

    P2 = U2[:, 0:m]
    return numpy.dot(P2.T, P).T

def apply_lda(U, D):
    return U.T @ D

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
    W = compute_lda_geig(D, L, m = 2)
    print(W)
    print(compute_lda_JointDiag(D, L, m=2)) 
   

    
