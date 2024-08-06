import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import fmin_l_bfgs_b
from pca import compute_pca, apply_pca
import os

# Load and preprocess the data
def load(pathname, visualization=0):
    df = pd.read_csv(pathname, header=None)
    if visualization:
        print(df.head())
    attribute = np.array(df.iloc[:, 0:len(df.columns) - 1]).T
    label = np.array(df.iloc[:, -1])
    return attribute, label

def split_db_2to1(D, L, seed=0):
    nTrain = int(D.shape[1] * 2.0 / 3.0)
    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]
    DTR = D[:, idxTrain]
    DVAL = D[:, idxTest]
    LTR = L[idxTrain]
    LVAL = L[idxTest]
    return (DTR, LTR), (DVAL, LVAL)

# Expand features quadratically
def quadratic_expansion(D):
    n = D.shape[0]
    D_quad = np.vstack([D] + [D[i] * D[j] for i in range(n) for j in range(i, n)])
    return D_quad

class logRegClass:
    def __init__(self, DTR, LTR, l, piT=0.5):
        self.DTR = DTR
        self.LTR = LTR
        self.l = l
        self.piT = piT

    def logreg_obj(self, v):
        w = v[:-1]
        b = v[-1]
        ZTR = 2 * self.LTR - 1
        S = (np.dot(w.T, self.DTR) + b).ravel()
        if self.piT == 0.5:
            weights = np.ones(self.LTR.size) / self.LTR.size
        else:
            weights = np.where(self.LTR == 1, self.piT / np.sum(self.LTR == 1), (1 - self.piT) / np.sum(self.LTR == 0))
        log_likelihood = np.sum(weights * np.logaddexp(0, -ZTR * S))
        J = (self.l / 2) * np.dot(w, w) + log_likelihood
        return J

# Function to train logistic regression
def train_logreg(DTR, LTR, l, prior_weighted=False, piT=0.5):
    v0 = np.zeros(DTR.shape[0] + 1)
    if prior_weighted:
        obj = logRegClass(DTR, LTR, l, piT)
    else:
        obj = logRegClass(DTR, LTR, l)
    v_opt, _, _ = fmin_l_bfgs_b(obj.logreg_obj, v0, approx_grad=True)
    return v_opt

def compute_confusion_matrix(predictions, labels, positive_label=1):
    TP = np.sum((predictions == positive_label) & (labels == positive_label))
    TN = np.sum((predictions != positive_label) & (labels != positive_label))
    FP = np.sum((predictions == positive_label) & (labels != positive_label))
    FN = np.sum((predictions != positive_label) & (labels == positive_label))
    return TP, TN, FP, FN

def compute_dcf(TP, TN, FP, FN, pi, Cfn, Cfp):
    Pfn = FN / (FN + TP) if (FN + TP) != 0 else 0
    Pfp = FP / (FP + TN) if (FP + TN) != 0 else 0
    dcf = pi * Cfn * Pfn + (1 - pi) * Cfp * Pfp
    return dcf

def minDCFcalc(pi, Cfn, Cfp, scores, labels):
    thresholds = np.sort(scores)
    min_dcf = float('inf')

    for threshold in thresholds:
        predictions = (scores >= threshold).astype(int)
        TP, TN, FP, FN = compute_confusion_matrix(predictions, labels)
        dcf = compute_dcf(TP, TN, FP, FN, pi, Cfn, Cfp)
        if dcf < min_dcf:
            min_dcf = dcf

    threshold = -np.log((pi * Cfn) / ((1 - pi) * Cfp))
    predictions = (scores >= threshold).astype(int)
    TP, TN, FP, FN = compute_confusion_matrix(predictions, labels)
    actual_dcf = compute_dcf(TP, TN, FP, FN, pi, Cfn, Cfp)

    return actual_dcf, min_dcf

# Function to compute metrics
def compute_metrics(v, DVAL, LVAL, pi_emp, piT):
    w, b = v[:-1], v[-1]
    scores = (np.dot(w.T, DVAL) + b).ravel()
    s_llr = scores - np.log(pi_emp / (1 - pi_emp))
    actualDCF, minDCF = minDCFcalc(piT, 1, 1, s_llr, LVAL)
    return actualDCF, minDCF

if __name__ == '__main__':
    path = 'Train.txt'
    D, L = load(path)
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)
    # Range of lambda values
    lambdas = np.logspace(-4, 2, 13)
    pi_emp = np.mean(LTR)
    piT = 0.1

    # full dataset
    actual_DCFs_full = []
    min_DCFs_full = []

    for l in lambdas:
        v_opt = train_logreg(DTR, LTR, l)
        actualDCF, minDCF = compute_metrics(v_opt, DVAL, LVAL, pi_emp, piT)
        actual_DCFs_full.append(actualDCF)
        min_DCFs_full.append(minDCF)
        print(f'Lambda: {l}, Actual DCF: {actualDCF}, Min DCF: {minDCF}')  # Debugging line

    # Plot full dataset
    plt.figure(figsize=(10, 6))
    plt.plot(lambdas, actual_DCFs_full, marker='o', label='Actual DCF (full)')
    plt.plot(lambdas, min_DCFs_full, marker='x', label='Min DCF (full)')
    plt.xscale('log')
    plt.xlabel('Lambda')
    plt.ylabel('DCF')
    plt.title('DCF vs. Lambda for Logistic Regression (Full Dataset)')
    plt.legend()
    plt.savefig(f"{os.getcwd()}/Image/Logistic_Regression.png")
    plt.grid(True)
    plt.show()

    print("Actual DCFs for different values of lambda (full):", actual_DCFs_full)
    print("Min DCFs for different values of lambda (full):", min_DCFs_full)

    # fewer training samples (1 out of 50)
    DTR_reduced = DTR[:, ::50]
    LTR_reduced = LTR[::50]

    actual_DCFs_reduced = []
    min_DCFs_reduced = []

    for l in lambdas:
        v_opt = train_logreg(DTR_reduced, LTR_reduced, l)
        actualDCF, minDCF = compute_metrics(v_opt, DVAL, LVAL, pi_emp, piT)
        actual_DCFs_reduced.append(actualDCF)
        min_DCFs_reduced.append(minDCF)

    # Plot reduced dataset
    plt.figure(figsize=(10, 6))
    plt.plot(lambdas, actual_DCFs_reduced, marker='o', label='Actual DCF (reduced)')
    plt.plot(lambdas, min_DCFs_reduced, marker='x', label='Min DCF (reduced)')
    plt.xscale('log')
    plt.xlabel('Lambda')
    plt.ylabel('DCF')
    plt.title('DCF vs. Lambda for Logistic Regression (Reduced Dataset)')
    plt.legend()
    plt.savefig(f"{os.getcwd()}/Image/Logistic_Regression_reduced_dataset.png")
    plt.grid(True)
    plt.show()

    print("Actual DCFs for different values of lambda (reduced):", actual_DCFs_reduced)
    print("Min DCFs for different values of lambda (reduced):", min_DCFs_reduced)

    # Prior-weighted logistic regression with full dataset
    actual_DCFs_prior_weighted = []
    min_DCFs_prior_weighted = []

    for l in lambdas:
        v_opt = train_logreg(DTR, LTR, l, prior_weighted=True, piT=piT)
        actualDCF, minDCF = compute_metrics(v_opt, DVAL, LVAL, pi_emp, piT)
        actual_DCFs_prior_weighted.append(actualDCF)
        min_DCFs_prior_weighted.append(minDCF)

    # Plot prior-weighted full dataset
    plt.figure(figsize=(10, 6))
    plt.plot(lambdas, actual_DCFs_prior_weighted, marker='o', label='Actual DCF (prior-weighted)')
    plt.plot(lambdas, min_DCFs_prior_weighted, marker='x', label='Min DCF (prior-weighted)')
    plt.xscale('log')
    plt.xlabel('Lambda')
    plt.ylabel('DCF')
    plt.title('DCF vs. Lambda for Prior-Weighted Logistic Regression (Full Dataset)')
    plt.legend()
    plt.savefig(f"{os.getcwd()}/Image/Prior_Weighted_QuadraticRegression.png")
    plt.grid(True)
    plt.show()

    print("Actual DCFs for different values of lambda (prior-weighted):", actual_DCFs_prior_weighted)
    print("Min DCFs for different values of lambda (prior-weighted):", min_DCFs_prior_weighted)

    # Quadratic logistic regression with full dataset
    DTR_quad = quadratic_expansion(DTR)
    DVAL_quad = quadratic_expansion(DVAL)
    actual_DCFs_quad = []
    min_DCFs_quad = []

    for l in lambdas:
        v_opt = train_logreg(DTR_quad, LTR, l)
        actualDCF, minDCF = compute_metrics(v_opt, DVAL_quad, LVAL, pi_emp, piT)
        actual_DCFs_quad.append(actualDCF)
        min_DCFs_quad.append(minDCF)

    # Plot quadratic logistic regression
    plt.figure(figsize=(10, 6))
    plt.plot(lambdas, actual_DCFs_quad, marker='o', label='Actual DCF (quadratic)')
    plt.plot(lambdas, min_DCFs_quad, marker='x', label='Min DCF (quadratic)')
    plt.xscale('log')
    plt.xlabel('Lambda')
    plt.ylabel('DCF')
    plt.title('DCF vs. Lambda for Quadratic Logistic Regression (Full Dataset)')
    plt.legend()
    plt.savefig(f"{os.getcwd()}/Image/QuadraticRegression.png")
    plt.grid(True)
    plt.show()

    print("Actual DCFs for different values of lambda (quadratic):", actual_DCFs_quad)
    print("Min DCFs for different values of lambda (quadratic):", min_DCFs_quad)

    # PCA logistic regression with full dataset
    m = 5 
    P = compute_pca(DTR, m)
    DTR_pca = apply_pca(P, DTR)
    DVAL_pca = apply_pca(P, DVAL)
    actual_DCFs_pca = []
    min_DCFs_pca = []

    for l in lambdas:
        v_opt = train_logreg(DTR_pca, LTR, l)
        actualDCF, minDCF = compute_metrics(v_opt, DVAL_pca, LVAL, pi_emp, piT)
        actual_DCFs_pca.append(actualDCF)
        min_DCFs_pca.append(minDCF)

    # Plot PCA logistic regression
    plt.figure(figsize=(10, 6))
    plt.plot(lambdas, actual_DCFs_pca, marker='o', label='Actual DCF (PCA)')
    plt.plot(lambdas, min_DCFs_pca, marker='x', label='Min DCF (PCA)')
    plt.xscale('log')
    plt.xlabel('Lambda')
    plt.ylabel('DCF')
    plt.title('DCF vs. Lambda for PCA Logistic Regression (Full Dataset)')
    plt.legend()
    plt.savefig(f"{os.getcwd()}/Image/LogisticRegression_PCA.png")
    plt.grid(True)
    plt.show()

    print("Actual DCFs for different values of lambda (PCA):", actual_DCFs_pca)
    print("Min DCFs for different values of lambda (PCA):", min_DCFs_pca)
