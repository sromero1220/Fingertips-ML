import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize
from bayesRisk import compute_minDCF_binary_fast, compute_actDCF_binary_fast
import os

def vcol(x):
    return x.reshape((x.size, 1))

def vrow(x):
    return x.reshape((1, x.size))

def load(filepath):
    df = pd.read_csv(filepath, header=None)
    attributes = np.array(df.iloc[:, :-1]).T
    labels = np.array(df.iloc[:, -1])
    return attributes, labels

def split_db_2to1(D, L, seed=0):
    nTrain = int(D.shape[1] * 2.0 / 3.0)
    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1])
    idxTrain = idx[:nTrain]
    idxTest = idx[nTrain:]
    DTR = D[:, idxTrain]
    DVAL = D[:, idxTest]
    LTR = L[idxTrain]
    LVAL = L[idxTest]
    return (DTR, LTR), (DVAL, LVAL)

def train_dual_SVM_linear(DTR, LTR, C, K=1):
    ZTR = LTR * 2.0 - 1.0
    DTR_EXT = np.vstack([DTR, np.ones((1, DTR.shape[1])) * K])
    H = np.dot(DTR_EXT.T, DTR_EXT) * vcol(ZTR) * vrow(ZTR)

    def fOpt(alpha):
        Ha = H @ vcol(alpha)
        loss = 0.5 * (vrow(alpha) @ Ha).ravel() - alpha.sum()
        grad = Ha.ravel() - np.ones(alpha.size)
        return loss, grad

    alphaStar, _, _ = scipy.optimize.fmin_l_bfgs_b(fOpt, np.zeros(DTR_EXT.shape[1]), bounds=[(0, C) for _ in LTR], factr=1.0)
    w_hat = (vrow(alphaStar) * vrow(ZTR) * DTR_EXT).sum(1)
    w, b = w_hat[0:DTR.shape[0]], w_hat[-1] * K
    return w, b

def polyKernel(degree, c):
    def polyKernelFunc(D1, D2):
        return (np.dot(D1.T, D2) + c) ** degree
    return polyKernelFunc

def rbfKernel(gamma):
    def rbfKernelFunc(D1, D2):
        D1Norms = (D1**2).sum(0)
        D2Norms = (D2**2).sum(0)
        Z = vcol(D1Norms) + vrow(D2Norms) - 2 * np.dot(D1.T, D2)
        return np.exp(-gamma * Z)
    return rbfKernelFunc

def train_dual_SVM_kernel(DTR, LTR, C, kernelFunc, eps=1.0):
    ZTR = LTR * 2.0 - 1.0
    K = kernelFunc(DTR, DTR) + eps
    H = vcol(ZTR) * vrow(ZTR) * K

    def fOpt(alpha):
        Ha = H @ vcol(alpha)
        loss = 0.5 * (vrow(alpha) @ Ha).ravel() - alpha.sum()
        grad = Ha.ravel() - np.ones(alpha.size)
        return loss, grad

    alphaStar, _, _ = scipy.optimize.fmin_l_bfgs_b(fOpt, np.zeros(DTR.shape[1]), bounds=[(0, C) for _ in LTR], factr=1.0)
    def fScore(DTE):
        K = kernelFunc(DTR, DTE) + eps
        H = vcol(alphaStar) * vcol(ZTR) * K
        return H.sum(0)
    return fScore

if __name__ == '__main__':
    path = 'Train.txt'
    D, L = load(path)
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)
    Cs = np.logspace(-5, 0, 11)
    gammas = np.exp(np.linspace(-4, -1, 4))
    piT = 0.1

    # Linear SVM
    actual_dcf_full_linear = []
    min_dcf_full_linear = []
    for C in Cs:
        w, b = train_dual_SVM_linear(DTR, LTR, C, K=1.0)
        SVAL = (vrow(w) @ DVAL + b).ravel()
        actual_dcf = compute_actDCF_binary_fast(SVAL, LVAL, piT, 1.0, 1.0)
        min_dcf = compute_minDCF_binary_fast(SVAL, LVAL, piT, 1.0, 1.0)
        actual_dcf_full_linear.append(actual_dcf)
        min_dcf_full_linear.append(min_dcf)
        #print(f'C: {C}, Actual DCF (Linear): {actual_dcf}, Min DCF (Linear): {min_dcf}')

    plt.figure(figsize=(10, 6))
    plt.plot(Cs, actual_dcf_full_linear, marker='o', label='Actual DCF (Linear)')
    plt.plot(Cs, min_dcf_full_linear, marker='x', label='Min DCF (Linear)')
    plt.xscale('log')
    plt.xlabel('C')
    plt.ylabel('DCF')
    plt.title('DCF vs. C for Linear SVM')
    plt.legend()
    plt.savefig(f"{os.getcwd()}/Image/Linear_SVM.png")
    plt.grid(True)
    plt.show()
    
    print("Best Actual DCF for Linear SVM: ", min(actual_dcf_full_linear))
    print("Best C for Linear SVM: ", Cs[np.argmin(actual_dcf_full_linear)])
    print("minDCF for Best Actual DCF Linear SVM: ", min_dcf_full_linear[np.argmin(actual_dcf_full_linear)])
    print("Best minDCF for Linear SVM: ", min(min_dcf_full_linear))
    

    # Polynomial SVM
    actual_dcf_full_poly = []
    min_dcf_full_poly = []
    poly_kernel = polyKernel(2, 1)
    for C in Cs:
        fScore = train_dual_SVM_kernel(DTR, LTR, C, poly_kernel, eps=0)
        SVAL = fScore(DVAL)
        actual_dcf = compute_actDCF_binary_fast(SVAL, LVAL, piT, 1.0, 1.0)
        min_dcf = compute_minDCF_binary_fast(SVAL, LVAL, piT, 1.0, 1.0)
        actual_dcf_full_poly.append(actual_dcf)
        min_dcf_full_poly.append(min_dcf)
        #print(f'C: {C}, Actual DCF (Poly): {actual_dcf}, Min DCF (Poly): {min_dcf}')

    plt.figure(figsize=(10, 6))
    plt.plot(Cs, actual_dcf_full_poly, marker='o', label='Actual DCF (Poly)')
    plt.plot(Cs, min_dcf_full_poly, marker='x', label='Min DCF (Poly)')
    plt.xscale('log')
    plt.xlabel('C')
    plt.ylabel('DCF')
    plt.title('DCF vs. C for Polynomial SVM')
    plt.legend()
    plt.savefig(f"{os.getcwd()}/Image/Polynomial_SVM.png")
    plt.grid(True)
    plt.show()
    
    print("Best Actual DCF for Polynomial SVM: ", min(actual_dcf_full_poly))
    print("Best C for Polynomial SVM: ", Cs[np.argmin(actual_dcf_full_poly)])
    print("minDCF for Best Actual DCF Polynomial SVM: ", min_dcf_full_poly[np.argmin(actual_dcf_full_poly)])
    print("Best minDCF for Polynomial SVM: ", min(min_dcf_full_poly))

    # RBF SVM
    Cs = np.logspace(-3, 2, 11)
    actual_dcf_full_rbf = {gamma: [] for gamma in gammas}
    min_dcf_full_rbf = {gamma: [] for gamma in gammas}
    for gamma in gammas:
        rbf_kernel = rbfKernel(gamma)
        for C in Cs:
            fScore = train_dual_SVM_kernel(DTR, LTR, C, rbf_kernel, eps=1.0)
            SVAL = fScore(DVAL)
            actual_dcf = compute_actDCF_binary_fast(SVAL, LVAL, piT, 1.0, 1.0)
            min_dcf = compute_minDCF_binary_fast(SVAL, LVAL, piT, 1.0, 1.0)
            actual_dcf_full_rbf[gamma].append(actual_dcf)
            min_dcf_full_rbf[gamma].append(min_dcf)
            #print(f'Gamma: {gamma}, C: {C}, Actual DCF (RBF): {actual_dcf}, Min DCF (RBF): {min_dcf}')

    plt.figure(figsize=(10, 6))
    for gamma in gammas:
        plt.plot(Cs, actual_dcf_full_rbf[gamma], marker='o', label=f'Actual DCF (RBF, gamma={gamma})')
        plt.plot(Cs, min_dcf_full_rbf[gamma], marker='x', label=f'Min DCF (RBF, gamma={gamma})')
    plt.xscale('log')
    plt.xlabel('C')
    plt.ylabel('DCF')
    plt.title('DCF vs. C for RBF SVM')
    plt.legend()
    plt.savefig(f"{os.getcwd()}/Image/RBF_SVM.png")
    plt.grid(True)        
    plt.show()
    
    print("Best Actual DCF for RBF SVM: ", min([min(actual_dcf_full_rbf[gamma]) for gamma in gammas]))
    print("Best gamma for RBF SVM: ", gammas[np.argmin([min(actual_dcf_full_rbf[gamma]) for gamma in gammas])])
    print("minDCF for Best Actual DCF RBF SVM: ", min([min_dcf_full_rbf[gamma][np.argmin(actual_dcf_full_rbf[gamma])] for gamma in gammas]))
    print("Best C for RBF SVM: ", Cs[np.argmin([min(actual_dcf_full_rbf[gamma]) for gamma in gammas])])
    print("Best minDCF for RBF SVM: ", min([min(min_dcf_full_rbf[gamma]) for gamma in gammas]))
    print("Gamma for Best minDCF RBF SVM: ", gammas[np.argmin([min(min_dcf_full_rbf[gamma]) for gamma in gammas])])
