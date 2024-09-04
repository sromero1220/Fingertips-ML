import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import fmin_l_bfgs_b
from pca import compute_pca, apply_pca
import os
from bayesRisk import compute_empirical_Bayes_risk_binary_llr_optimal_decisions as compute_actualDCF
from bayesRisk import compute_minDCF_binary_fast as compute_minDCF

def vcol(x):
    return x.reshape((x.size, 1))

def vrow(x):
    return x.reshape((1, x.size))

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

def load(pathname, visualization=0):
    df = pd.read_csv(pathname, header=None)
    if visualization:
        print(df.head())
    attribute = np.array(df.iloc[:, 0:len(df.columns) - 1]).T
    label = np.array(df.iloc[:, -1])
    return attribute, label

# Quadratic expansion of features
def quadratic_expansion(D):
    n = D.shape[0]
    D_quad = np.vstack([D] + [D[i] * D[j] for i in range(n) for j in range(i, n)])
    return D_quad

def trainLogRegBinary(DTR, LTR, l):

    ZTR = LTR * 2.0 - 1.0

    def logreg_obj_with_grad(v):
        w = v[:-1]
        b = v[-1]
        s = np.dot(vcol(w).T, DTR).ravel() + b

        loss = np.logaddexp(0, -ZTR * s)

        G = -ZTR / (1.0 + np.exp(ZTR * s))
        GW = (vrow(G) * DTR).mean(1) + l * w.ravel()
        Gb = G.mean()
        return loss.mean() + l / 2 * np.linalg.norm(w)**2, np.hstack([GW, np.array(Gb)])

    vf = fmin_l_bfgs_b(logreg_obj_with_grad, x0=np.zeros(DTR.shape[0]+1))[0]
    #print("Log-reg - lambda = %e - J*(w, b) = %e" % (l, logreg_obj_with_grad(vf)[0]))
    return vf[:-1], vf[-1]

def trainWeightedLogRegBinary(DTR, LTR, l, pT):

    ZTR = LTR * 2.0 - 1.0
    
    wTrue = pT / (ZTR > 0).sum()
    wFalse = (1 - pT) / (ZTR < 0).sum()

    def logreg_obj_with_grad(v):
        w = v[:-1]
        b = v[-1]
        s = np.dot(vcol(w).T, DTR).ravel() + b

        loss = np.logaddexp(0, -ZTR * s)
        loss[ZTR > 0] *= wTrue
        loss[ZTR < 0] *= wFalse

        G = -ZTR / (1.0 + np.exp(ZTR * s))
        G[ZTR > 0] *= wTrue
        G[ZTR < 0] *= wFalse
        
        GW = (vrow(G) * DTR).sum(1) + l * w.ravel()
        Gb = G.sum()
        return loss.sum() + l / 2 * np.linalg.norm(w)**2, np.hstack([GW, np.array(Gb)])

    vf = fmin_l_bfgs_b(logreg_obj_with_grad, x0=np.zeros(DTR.shape[0]+1))[0]
    #print("Weighted Log-reg (pT %e) - lambda = %e - J*(w, b) = %e" % (pT, l, logreg_obj_with_grad(vf)[0]))
    return vf[:-1], vf[-1]


def compute_metrics(w, b, DVAL, LVAL, pi_emp, piT):
    scores = (np.dot(w.T, DVAL) + b).ravel()
    s_llr = scores - np.log(pi_emp / (1 - pi_emp))
    actualDCF = compute_actualDCF(s_llr, LVAL, piT, 1, 1, normalize=True)
    minDCF = compute_minDCF(s_llr, LVAL, piT, 1, 1, returnThreshold=False)
    return actualDCF, minDCF

if __name__ == '__main__':
    path = 'Train.txt'
    D, L = load(path)
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)
    
    lambdas = np.logspace(-4, 2, 13)# Range of lambda values
    pi_emp = (LTR == 1).sum() / LTR.size
    piT = 0.1

    # full dataset
    actual_DCFs_full = []
    min_DCFs_full = []

    for l in lambdas:
        w, b = trainLogRegBinary(DTR, LTR, l)
        actualDCF, minDCF = compute_metrics(w, b, DVAL, LVAL, pi_emp, piT)
        actual_DCFs_full.append(actualDCF)
        min_DCFs_full.append(minDCF)

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

    Best_lambda = lambdas[np.argmin(actual_DCFs_full)]
    print("Best Actual DCF for different values of lambda (full):", min(actual_DCFs_full))
    print("min DCF for Best Actual DCF different values of lambda (full):", min_DCFs_full[np.argmin(actual_DCFs_full)])
    print("Best lambda for different values of lambda (full):", Best_lambda)
    print("Best Min DCF for different values of lambda (full):", min(min_DCFs_full))

    # fewer training samples (1 out of 50)
    DTR_reduced = DTR[:, ::50]
    LTR_reduced = LTR[::50]

    actual_DCFs_reduced = []
    min_DCFs_reduced = []

    for l in lambdas:
        w, b = trainLogRegBinary(DTR_reduced, LTR_reduced, l)
        actualDCF, minDCF = compute_metrics(w, b, DVAL, LVAL, pi_emp, piT)
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

    Best_lambda_reduced = lambdas[np.argmin(actual_DCFs_reduced)]
    print("Best Actual DCF for different values of lambda (reduced):", min(actual_DCFs_reduced))
    print("min DCF for Best Actual DCF different values of lambda (reduced):", min_DCFs_reduced[np.argmin(actual_DCFs_reduced)])
    print("Best lambda for different values of lambda (reduced):", Best_lambda_reduced)
    print("Best Min DCF for different values of lambda (reduced):", min(min_DCFs_reduced))

    # Prior-weighted logistic regression with full dataset
    actual_DCFs_prior_weighted = []
    min_DCFs_prior_weighted = []

    for l in lambdas:
        w, b = trainWeightedLogRegBinary(DTR, LTR, l, piT)
        actualDCF, minDCF = compute_metrics(w, b, DVAL, LVAL, pi_emp, piT)
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

    Best_lambda_prior_weighted = lambdas[np.argmin(actual_DCFs_prior_weighted)] 
    print("Best Actual DCF for different values of lambda (prior-weighted):", min(actual_DCFs_prior_weighted))
    print("min DCF for Best Actual DCF different values of lambda (prior-weighted):", min_DCFs_prior_weighted[np.argmin(actual_DCFs_prior_weighted)])
    print("Best lambda for different values of lambda (prior-weighted):", Best_lambda_prior_weighted)
    print("Best Min DCFs for different values of lambda (prior-weighted):", min(min_DCFs_prior_weighted))

    # Quadratic logistic regression with full dataset
    DTR_quad = quadratic_expansion(DTR)
    DVAL_quad = quadratic_expansion(DVAL)
    actual_DCFs_quad = []
    min_DCFs_quad = []

    for l in lambdas:
        w, b = trainLogRegBinary(DTR_quad, LTR, l)
        actualDCF, minDCF = compute_metrics(w, b, DVAL_quad, LVAL, pi_emp, piT)
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
    
    
    Best_lambda_quad = lambdas[np.argmin(actual_DCFs_quad)] 
    print("Best Actual DCF for different values of lambda (quadratic):", min(actual_DCFs_quad))
    print("min DCF for Best Actual DCF different values of lambda (quadratic):", min_DCFs_quad[np.argmin(actual_DCFs_quad)])
    print("Best lambda for different values of lambda (quadratic):", Best_lambda_quad)  
    print("Best Min DCF for different values of lambda (quadratic):", min(min_DCFs_quad))

    # PCA logistic regression with full dataset
    m = 5 
    P = compute_pca(DTR, m)
    DTR_pca = apply_pca(P, DTR)
    DVAL_pca = apply_pca(P, DVAL)
    actual_DCFs_pca = []
    min_DCFs_pca = []

    for l in lambdas:
        w, b = trainLogRegBinary(DTR_pca, LTR, l)
        actualDCF, minDCF = compute_metrics(w, b, DVAL_pca, LVAL, pi_emp, piT)
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
    
    Best_lambda_pca = lambdas[np.argmin(actual_DCFs_pca)]
    print("Best Actual DCF for different values of lambda (PCA):", min(actual_DCFs_pca))
    print("min DCF for Best Actual DCF different values of lambda (PCA):", min_DCFs_pca[np.argmin(actual_DCFs_pca)])
    print("Best lambda for different values of lambda (PCA):", Best_lambda_pca)
    print("Best Min DCF for different values of lambda (PCA):", min(min_DCFs_pca))
