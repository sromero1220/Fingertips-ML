import pca
import lda
import numpy
import pandas as pd
import os
import matplotlib.pyplot as plt

from lda import vcol, vrow

def split_db_2to1(D, L, seed=0):
    
    nTrain = int(D.shape[1]*2.0/3.0)
    numpy.random.seed(seed)
    idx = numpy.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]
    
    DTR = D[:, idxTrain]
    DVAL = D[:, idxTest]
    LTR = L[idxTrain]
    LVAL = L[idxTest]
    
    return (DTR, LTR), (DVAL, LVAL)

def load(pathname, visualization=0):
    df = pd.read_csv(pathname, header=None)
    if visualization:
        print(df.head())
    attribute = numpy.array(df.iloc[:, 0 : len(df.columns) - 1]).T
    label = numpy.array(df.iloc[:, -1])
    return attribute, label

if __name__ == '__main__':
    # numpy.set_printoptions(threshold=numpy.inf)
    path = os.path.abspath("Train.txt")
    D, L = load(path)
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)
    # Solution without PCA pre-processing and threshold selection. The threshold is chosen half-way between the two classes
    ULDA = lda.compute_lda_JointDiag(DTR, LTR, m=1)

    DTR_lda = lda.apply_lda(ULDA, DTR)

    # Check if the Fake fingerprint class samples are on the right of the genuine fingerprints samples on the training set. If not, we reverse ULDA and re-apply the transformation.
    if DTR_lda[0, LTR==1].mean() > DTR_lda[0, LTR==0].mean():
        ULDA = -ULDA
        DTR_lda = lda.apply_lda(ULDA, DTR)

    DVAL_lda  = lda.apply_lda(ULDA, DVAL)

    threshold = (DTR_lda[0, LTR==0].mean() + DTR_lda[0, LTR==1].mean()) / 2.0 # Estimated only on model training data
    print('Threshold:', threshold)
    PVAL = numpy.zeros(shape=LVAL.shape, dtype=numpy.int32)
    PVAL[DVAL_lda[0] >= threshold] = 0
    PVAL[DVAL_lda[0] < threshold] = 1
    print('Solution without PCA pre-processing and threshold selection')
    print('Labels:     ', LVAL)
    print('Predictions:', PVAL)
    print('Number of erros:', (PVAL != LVAL).sum(), '(out of %d samples)' % (LVAL.size))
    print('Error rate: %.1f%%' % ( (PVAL != LVAL).sum() / float(LVAL.size) *100 ))
          
    # Solution with PCA pre-processing with dimension m.
    m = 5
    UPCA = pca.compute_pca(DTR, m = m) # Estimated only on model training data
    DTR_pca = pca.apply_pca(UPCA, DTR)   # Applied to original model training data
    DVAL_pca = pca.apply_pca(UPCA, DVAL) # Applied to original validation data

    ULDA = lda.compute_lda_JointDiag(DTR_pca, LTR, m = 1) # Estimated only on model training data, after PCA has been applied

    DTR_lda = lda.apply_lda(ULDA, DTR_pca)   # Applied to PCA-transformed model training data, the projected training samples are required to check the orientation of the direction and to compute the threshold
    # Check if the Fake fingerprint class samples are, on average, on the right of the genuine fingerprints samples on the training set. If not, we reverse ULDA and re-apply the transformation
    if DTR_lda[0, LTR==1].mean() > DTR_lda[0, LTR==0].mean():
        ULDA = -ULDA
        DTR_lda = lda.apply_lda(ULDA, DTR_pca)

    DVAL_lda = lda.apply_lda(ULDA, DVAL_pca) # Applied to PCA-transformed validation data
    
    PVAL = numpy.zeros(shape=LVAL.shape, dtype=numpy.int32)
    PVAL[DVAL_lda[0] >= threshold] = 0
    PVAL[DVAL_lda[0] < threshold] = 1
    print('Solution with PCA pre-processing with dimension m = %d' % m)
    print('Labels:     ', LVAL)
    print('Predictions:', PVAL)
    print('Number of erros:', (PVAL != LVAL).sum(), '(out of %d samples)' % (LVAL.size))
    print('Error rate: %.1f%%' % ( (PVAL != LVAL).sum() / float(LVAL.size) *100 ))
    
    
