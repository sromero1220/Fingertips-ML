import numpy as np
import scipy.linalg
import pandas as pd
import os

def load(pathname, visualization=0):
    df = pd.read_csv(pathname, header=None)
    if visualization:
        print(df.head())
    attribute = np.array(df.iloc[:, 0 : len(df.columns) - 1]).T
    label = np.array(df.iloc[:, -1])
    return attribute, label
def vcol(x):
    return x.reshape((x.size, 1))

def vrow(x):
    return x.reshape((1, x.size))

def split_db_2to1(D, L, seed=0):
    
    nTrain = int(D.shape[1]*2.0/3.0)
    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]
    
    DTR = D[:, idxTrain]
    DVAL = D[:, idxTest]
    LTR = L[idxTrain]
    LVAL = L[idxTest]
    
    return (DTR, LTR), (DVAL, LVAL)

def compute_mu_C(D):
    mu = vcol(D.mean(1))
    C = ((D-mu) @ (D-mu).T) / float(D.shape[1])
    return mu, C

def logpdf_GAU_ND(x, mu, C):
    P = np.linalg.inv(C)
    return -0.5*x.shape[0]*np.log(np.pi*2) - 0.5*np.linalg.slogdet(C)[1] - 0.5 * ((x-mu) * (P @ (x-mu))).sum(0)


def Gau_MVG_ML_estimates(D, L):
    labelSet = set(L)
    hParams = {}
    for l in labelSet:
        DX = D[:, L==l]
        hParams[l] = compute_mu_C(DX)
    return hParams

#full covariance matrix and then extract the diagonal (diagonal of the covariance matrix)
def Gau_Naive_ML_estimates(D, L):
    labelSet = set(L)
    hParams = {}
    for lab in labelSet:
        DX = D[:, L==lab]
        mu, C = compute_mu_C(DX)
        hParams[lab] = (mu, C * np.eye(D.shape[0]))
    return hParams

# Within-class covairance matrix is a weighted mean of the covraince matrices of the different classes
def Gau_Tied_ML_estimates(D, L):
    labelSet = set(L)
    hParams = {}
    hMeans = {}
    CGlobal = 0
    for lab in labelSet:
        DX = D[:, L==lab]
        mu, C_class = compute_mu_C(DX)
        CGlobal += C_class * DX.shape[1]
        hMeans[lab] = mu
    CGlobal = CGlobal / D.shape[1]
    for lab in labelSet:
        hParams[lab] = (hMeans[lab], CGlobal)
    return hParams

# log-densities
def compute_log_likelihood_Gau(D, hParams):

    S = np.zeros((len(hParams), D.shape[1]))
    for lab in range(S.shape[0]):
        S[lab, :] = logpdf_GAU_ND(D, hParams[lab][0], hParams[lab][1])
    return S

# log-postorior matrix from log-likelihood matrix and prior array
def compute_logPosterior(S_logLikelihood, v_prior):
    SJoint = S_logLikelihood + vcol(np.log(v_prior))
    SMarginal = vrow(scipy.special.logsumexp(SJoint, axis=0))
    SPost = SJoint - SMarginal
    return SPost
                     

# Correlation matrix calculation for the MVG model
def compute_correlation_matrix(C):
    std_dev = np.sqrt(np.diag(C))
    return C / np.outer(std_dev, std_dev)


def create_confusion_matrix(predictions, labels, num_classes):
    CM = np.array([0] * num_classes**2).reshape(num_classes, num_classes)
    for i in range(predictions.size):
        CM[predictions[i], round(labels[i])] += 1
    return CM
def calculate_minimum_detection_cost(pi, cfn, cfp, scores, labels):
    thresholds = np.append(np.min(scores) - 1, scores)
    thresholds = np.append(thresholds, np.max(scores) + 1)
    thresholds.sort()
    DCFdummy = min(pi*cfn, (1-pi)*cfp)
    minDCF = 100000
    for t in thresholds:
        predictions = [1 if el > t else 0 for el in scores]
        CM = create_confusion_matrix(np.array(predictions), labels, 2)
        fnr = CM[0, 1] / (CM[1, 1] + CM[0, 1])
        fpr = CM[1, 0] / (CM[0, 0] + CM[1, 0])
        DCFu = pi*cfn*fnr + (1-pi)*cfp*fpr
        minDCF = (DCFu / DCFdummy) if (DCFu / DCFdummy) < minDCF else minDCF

    return minDCF

if __name__ == '__main__':
    path = os.path.abspath("Train.txt")
    D, L = load(path)
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)


    hParams_MVG = Gau_MVG_ML_estimates(DTR, LTR)

    # Compute log-likelihood ratios for class 1 vs class 0
    LLR = logpdf_GAU_ND(DVAL, hParams_MVG[1][0], hParams_MVG[1][1]) - logpdf_GAU_ND(DVAL, hParams_MVG[0][0], hParams_MVG[0][1])

    # Predictions based on LLR with uniform class priors
    threshold = 0
    predictions = np.where(LLR >= threshold, 1, 0)
    error_rate_MVG = np.mean(predictions != LVAL)
    print(f"MVG Model Error Rate: {error_rate_MVG * 100:.2f}%")

    # Tied Gaussian model
    hParams_Tied = Gau_Tied_ML_estimates(DTR, LTR)

    # Compute LLR using the tied model
    LLR_tied = logpdf_GAU_ND(DVAL, hParams_Tied[1][0], hParams_Tied[1][1]) - logpdf_GAU_ND(DVAL, hParams_Tied[0][0], hParams_Tied[0][1])
    predictions_tied = np.where(LLR_tied >= threshold, 1, 0)
    error_rate_Tied = np.mean(predictions_tied != LVAL)
    print(f"Tied Gaussian Model Error Rate: {error_rate_Tied * 100:.2f}%")

    # Naive Bayes Gaussian model
    hParams_Naive = Gau_Naive_ML_estimates(DTR, LTR)

    # Compute LLR using the Naive Bayes model
    LLR_naive = logpdf_GAU_ND(DVAL, hParams_Naive[1][0], hParams_Naive[1][1]) - logpdf_GAU_ND(DVAL, hParams_Naive[0][0], hParams_Naive[0][1])
    predictions_naive = np.where(LLR_naive >= threshold, 1, 0)
    error_rate_Naive = np.mean(predictions_naive != LVAL)
    print(f"Naive Bayes Gaussian Model Error Rate: {error_rate_Naive * 100:.2f}%")

    # Print covariance and correlation matrices for each class
    for label, params in hParams_MVG.items():
        mu, C = params
        print(f"Class {label} - Covariance Matrix:\n{C}")
        correlation_matrix = compute_correlation_matrix(C)
        print(f"Class {label} - Correlation Matrix:\n{correlation_matrix}")
