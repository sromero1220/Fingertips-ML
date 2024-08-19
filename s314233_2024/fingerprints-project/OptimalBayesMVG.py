import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from pca import compute_pca, apply_pca

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

def Gau_Naive_ML_estimates(D, L):
    labelSet = set(L)
    hParams = {}
    for lab in labelSet:
        DX = D[:, L==lab]
        mu, C = compute_mu_C(DX)
        hParams[lab] = (mu, np.diag(np.diag(C)))
    return hParams

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

def compute_log_likelihood_Gau(D, hParams):
    S = np.zeros((len(hParams), D.shape[1]))
    for lab in range(S.shape[0]):
        S[lab, :] = logpdf_GAU_ND(D, hParams[lab][0], hParams[lab][1])
    return S
    
    
def bayes_error_plot(llratios, labels, title, pca=False):
    effPriorLogOdds = np.linspace(-4, 4, 21)
    pis = [1 / (1 + np.exp(- p)) for p in effPriorLogOdds]

    dcf = [compute_DCF_binary(pi, 1, 1, llratios, labels) for pi in pis]
    actualDCF_minDCF  = [minDCFcalc(pi, 1, 1, llratios, labels) for pi in pis]
    
    actualDCF = [x[0] for x in actualDCF_minDCF]
    minDCF = [x[1] for x in actualDCF_minDCF]

    plt.plot(effPriorLogOdds, dcf, label='DCF', color='r')
    plt.plot(effPriorLogOdds, minDCF, label='min DCF', color='b')
    plt.ylim([0, 1.1])
    plt.xlim([-4, 4])
    plt.xlabel('Effective Prior Log Odds')
    plt.ylabel('Detection Cost Function')
    plt.title(f'Bayes Error Plot ({title})')
    plt.legend()
    plt.savefig(f"{os.getcwd()}/Image/{title}_PCA_{pca}_Bayes_error.png")
    plt.show()
def compute_DCF_binary(pi, cfn, cfp, llratios, labels):
    CM, _ = optimal_bayes_decisions_binary(pi, cfn, cfp, llratios, labels)
    fnr = CM[0, 1] / (CM[1, 1] + CM[0, 1])
    fpr = CM[1, 0] / (CM[0, 0] + CM[1, 0])
    DCFu = pi*cfn*fnr + (1-pi)*cfp*fpr
    DCFdummy = min(pi*cfn, (1-pi)*cfp)
    return DCFu / DCFdummy
def optimal_bayes_decisions_binary(pi, cfn, cfp, llratios, labels):
    t = -np.log(pi * cfn / ((1 - pi) * cfp))
    predictions = [1 if el > t else 0 for el in llratios]
    return (confusion_matrix(np.array(predictions), labels, 2), t)
def confusion_matrix(predictions, labels, nclasses):
    CM = np.array([0] * nclasses**2).reshape(nclasses, nclasses)
    for i in range(predictions.size):
        CM[predictions[i], round(labels[i])] += 1
    return CM
def minDCFcalc(pi, cfn, cfp, scores, labels):
    thresholds = np.append(np.min(scores) - 1, scores)
    thresholds = np.append(thresholds, np.max(scores) + 1)
    thresholds.sort()
    DCFdummy = min(pi*cfn, (1-pi)*cfp)
    minDCF = 100000
    for t in thresholds:
        predictions = [1 if el > t else 0 for el in scores]
        CM = confusion_matrix(np.array(predictions), labels, 2)
        fnr = CM[0, 1] / (CM[1, 1] + CM[0, 1])
        fpr = CM[1, 0] / (CM[0, 0] + CM[1, 0])
        DCFu = pi*cfn*fnr + (1-pi)*cfp*fpr
        actualDCF= DCFu
        if (DCFu / DCFdummy) < minDCF:
            minDCF = (DCFu / DCFdummy) 
            actualDCF = DCFu 
        else: 
            minDCF=minDCF
            actualDCF=actualDCF

    return actualDCF, minDCF



def analyze_performance(D, L, applications, implement_pca=False, m=2):
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)
    if implement_pca:
        P = compute_pca(DTR, m)
        DTR = apply_pca(P, DTR)
        DVAL = apply_pca(P, DVAL)

    classifiers = {
        "MVG": Gau_MVG_ML_estimates,
        "Naive": Gau_Naive_ML_estimates,
        "Tied": Gau_Tied_ML_estimates
    }

    for name, clf in classifiers.items():
        hParams = clf(DTR, LTR)
        S_logLikelihood = compute_log_likelihood_Gau(DVAL, hParams)
        LLR = S_logLikelihood[1] - S_logLikelihood[0]
        print(f"\n{name} Classifier")

        for pi, cfn, cfp in applications:
            actualDCF, minDCF = minDCFcalc(pi, cfn, cfp, LLR, LVAL)
            normalizeDCF = compute_DCF_binary(pi, 1, 1, LLR, LVAL)
            print(f"Application (pi={pi}, cfn={cfn}, cfp={cfp}) -> DCF: {normalizeDCF:.3f}, min DCF: {minDCF:.3f}")

            if pi == 0.1 and cfn == 1.0 and cfp == 1.0:
                bayes_error_plot(LLR, LVAL, name, pca=implement_pca)

def find_best_pca_setup(D, L):
    best_dcf = float('inf')
    best_m = None
    applications = [(0.1, 1.0, 1.0)]
    for m in [2, 3, 4, 5, 6]:
        print(f"Testing PCA with m={m}")
        (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)
        P = compute_pca(DTR, m)
        DTR = apply_pca(P, DTR)
        DVAL = apply_pca(P, DVAL)
        hParams = Gau_MVG_ML_estimates(DTR, LTR)
        LLR = compute_log_likelihood_Gau(DVAL, hParams)[1] - compute_log_likelihood_Gau(DVAL, hParams)[0]
        for pi, cfn, cfp in applications:
            _, minDCF = minDCFcalc(pi, cfn, cfp, LLR, LVAL)
            if minDCF < best_dcf:
                best_dcf = minDCF
                best_m = m
    print(f"Best PCA setup: m={best_m} with DCF={best_dcf:.3f}")
    return best_m

if __name__ == '__main__':
    path = os.path.abspath("Train.txt")
    D, L = load(path)

    # Find the best PCA setup for π = 0.1 configuration
    best_m = find_best_pca_setup(D, L)
    
    applications = [
        (0.5, 1.0, 1.0),
        (0.9, 1.0, 1.0),
        (0.1, 1.0, 1.0),
        (0.5, 1.0, 9.0),
        (0.5, 9.0, 1.0)
    ]

    # Analyze performance using the original features
    analyze_performance(D, L, applications, implement_pca=False, m=best_m)

    # Analyze performance using the best PCA setup
    print("\n\nPCA analysis")
    analyze_performance(D, L, applications, implement_pca=True, m=best_m)
