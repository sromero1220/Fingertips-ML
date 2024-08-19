import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import scipy.special
from bayesRisk import compute_minDCF_binary_fast, compute_actDCF_binary_fast

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

def logpdf_GAU_ND(x, mu, C):
    P = np.linalg.inv(C)
    return -0.5*x.shape[0]*np.log(2 * np.pi) - 0.5*np.linalg.slogdet(C)[1] - 0.5 * ((x - mu) * (P @ (x - mu))).sum(0)

def logpdf_GMM(X, gmm):
    S = []
    for w, mu, C in gmm:
        logpdf_conditional = logpdf_GAU_ND(X, mu, C)
        logpdf_joint = logpdf_conditional + np.log(w)
        S.append(logpdf_joint)
    S = np.vstack(S)
    logdens = scipy.special.logsumexp(S, axis=0)
    return logdens

def smooth_covariance_matrix(C, psi):
    U, s, Vh = np.linalg.svd(C)
    s[s < psi] = psi
    CUpd = U @ (vcol(s) * U.T)
    return CUpd

def train_GMM_EM_Iteration(X, gmm, covType='full', psiEig=None):
    assert (covType.lower() in ['full', 'diagonal', 'tied'])
    S = []
    for w, mu, C in gmm:
        logpdf_conditional = logpdf_GAU_ND(X, mu, C)
        logpdf_joint = logpdf_conditional + np.log(w)
        S.append(logpdf_joint)
    S = np.vstack(S)
    logdens = scipy.special.logsumexp(S, axis=0)
    gammaAllComponents = np.exp(S - logdens)
    gmmUpd = []
    for gIdx in range(len(gmm)):
        gamma = gammaAllComponents[gIdx]
        Z = gamma.sum()
        F = vcol((vrow(gamma) * X).sum(1))
        S = (vrow(gamma) * X) @ X.T
        muUpd = F / Z
        CUpd = S / Z - muUpd @ muUpd.T
        wUpd = Z / X.shape[1]
        if covType.lower() == 'diagonal':
            CUpd = CUpd * np.eye(X.shape[0])
        gmmUpd.append((wUpd, muUpd, CUpd))
    if covType.lower() == 'tied':
        CTied = 0
        for w, mu, C in gmmUpd:
            CTied += w * C
        gmmUpd = [(w, mu, CTied) for w, mu, C in gmmUpd]
    if psiEig is not None:
        gmmUpd = [(w, mu, smooth_covariance_matrix(C, psiEig)) for w, mu, C in gmmUpd]
    return gmmUpd

def train_GMM_EM(X, gmm, covType='full', psiEig=None, epsLLAverage=1e-6, verbose=True):
    llOld = logpdf_GMM(X, gmm).mean()
    llDelta = None
    if verbose:
        print('GMM - it %3d - average ll %.8e' % (0, llOld))
    it = 1
    while (llDelta is None or llDelta > epsLLAverage):
        gmmUpd = train_GMM_EM_Iteration(X, gmm, covType=covType, psiEig=psiEig)
        llUpd = logpdf_GMM(X, gmmUpd).mean()
        llDelta = llUpd - llOld
        if verbose:
            print('GMM - it %3d - average ll %.8e' % (it, llUpd))
        gmm = gmmUpd
        llOld = llUpd
        it = it + 1
    if verbose:
        print('GMM - it %3d - average ll %.8e (eps = %e)' % (it, llUpd, epsLLAverage))
    return gmm

def split_GMM_LBG(gmm, alpha=0.1, verbose=True):
    gmmOut = []
    if verbose:
        print('LBG - going from %d to %d components' % (len(gmm), len(gmm) * 2))
    for (w, mu, C) in gmm:
        U, s, Vh = np.linalg.svd(C)
        d = U[:, 0:1] * s[0]**0.5 * alpha
        gmmOut.append((0.5 * w, mu - d, C))
        gmmOut.append((0.5 * w, mu + d, C))
    return gmmOut

def train_GMM_LBG_EM(X, numComponents, covType='full', psiEig=None, epsLLAverage=1e-6, lbgAlpha=0.1, verbose=True):
    mu, C = vcol(X.mean(1)), np.cov(X)
    if covType.lower() == 'diagonal':
        C = C * np.eye(X.shape[0])
    if psiEig is not None:
        gmm = [(1.0, mu, smooth_covariance_matrix(C, psiEig))]
    else:
        gmm = [(1.0, mu, C)]
    while len(gmm) < numComponents:
        if verbose:
            print('Average ll before LBG: %.8e' % logpdf_GMM(X, gmm).mean())
        gmm = split_GMM_LBG(gmm, lbgAlpha, verbose=verbose)
        if verbose:
            print('Average ll after LBG: %.8e' % logpdf_GMM(X, gmm).mean())
        gmm = train_GMM_EM(X, gmm, covType=covType, psiEig=psiEig, verbose=verbose, epsLLAverage=epsLLAverage)
    return gmm

def evaluate_gmm_performance(DTR, LTR, DVAL, LVAL, num_components_list, cov_types):
    results = {cov_type: [] for cov_type in cov_types}

    for cov_type in cov_types:
        for num_components in num_components_list:
            gmm0 = train_GMM_LBG_EM(DTR[:, LTR == 0], num_components, covType=cov_type, verbose=False, psiEig=0.000001)
            gmm1 = train_GMM_LBG_EM(DTR[:, LTR == 1], num_components, covType=cov_type, verbose=False, psiEig=0.000001)

            SLLR = logpdf_GMM(DVAL, gmm1) - logpdf_GMM(DVAL, gmm0)

            minDCF = compute_minDCF_binary_fast(SLLR, LVAL, 0.1, 1.0, 1.0)
            actDCF = compute_actDCF_binary_fast(SLLR, LVAL, 0.1, 1.0, 1.0)

            results[cov_type].append((num_components, minDCF, actDCF))
            print(f'numC = {num_components}, covType = {cov_type}, minDCF = {minDCF:.4f}, actDCF = {actDCF:.4f}')
    
    return results

def plot_results(results, num_components_list):
    plt.figure(figsize=(10, 6))
    for cov_type, values in results.items():
        num_components, minDCFs, actDCFs = zip(*values)
        plt.plot(num_components_list, minDCFs, marker='o', label=f'minDCF ({cov_type})')
        plt.plot(num_components_list, actDCFs, marker='x', label=f'actDCF ({cov_type})')
    plt.xscale('log')
    plt.xlabel('Number of Components')
    plt.ylabel('DCF')
    plt.title('DCF vs. Number of Components for GMM')
    plt.legend()
    plt.savefig(f"{os.getcwd()}/Image/GMM.png")
    plt.grid(True)
    plt.show()



if __name__ == '__main__':
    path = 'Train.txt'  
    D, L = load(path)
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)

    num_components_list = [1, 2, 4, 8, 16, 32]
    cov_types = ['full', 'diagonal']

    results = evaluate_gmm_performance(DTR, LTR, DVAL, LVAL, num_components_list, cov_types)

    plot_results(results, num_components_list)
    print("Best actDCF for Full Covariance: ", min(results['full'], key=lambda x: x[2])[2])
    print("Number of Components for Best actDCF Full Covariance: ", min(results['full'], key=lambda x: x[2])[0])
    print("minDCF for Best actDCF Full Covariance: ", min(results['full'], key=lambda x: x[2])[1])
    print("Best minDCF for Full Covariance: ", min(results['full'], key=lambda x: x[1]))

    
    print("Best actDCF for Diagonal Covariance: ", min(results['diagonal'], key=lambda x: x[2])[2])
    print("Number of Components for Best actDCF Diagonal Covariance: ", min(results['diagonal'], key=lambda x: x[2])[0])
    print("minDCF for Best actDCF Diagonal Covariance: ", min(results['diagonal'], key=lambda x: x[2])[1])
    print("Best minDCF for Diagonal Covariance: ", min(results['diagonal'], key=lambda x: x[1]))
   
    
    
