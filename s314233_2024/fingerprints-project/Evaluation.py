import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.optimize import fmin_l_bfgs_b

# Load the provided scripts' content as functions here
def vcol(x):
    return x.reshape((x.size, 1))

def vrow(x):
    return x.reshape((1, x.size))

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

def load_iris_binary():
    from sklearn.datasets import load_iris
    D, L = load_iris()['data'].T, load_iris()['target']
    D = D[:, L != 0]
    L = L[L != 0]
    L[L == 2] = 0
    return D, L

# Logistic Regression
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
    vf = fmin_l_bfgs_b(logreg_obj_with_grad, x0 = np.zeros(DTR.shape[0]+1))[0]
    return vf[:-1], vf[-1]

# SVM
def train_dual_SVM_linear(DTR, LTR, C, K=1):
    ZTR = LTR * 2.0 - 1.0
    DTR_EXT = np.vstack([DTR, np.ones((1, DTR.shape[1])) * K])
    H = np.dot(DTR_EXT.T, DTR_EXT) * vcol(ZTR) * vrow(ZTR)
    def fOpt(alpha):
        Ha = H @ vcol(alpha)
        loss = 0.5 * (vrow(alpha) @ Ha).ravel() - alpha.sum()
        grad = Ha.ravel() - np.ones(alpha.size)
        return loss, grad
    alphaStar, _, _ = fmin_l_bfgs_b(fOpt, np.zeros(DTR_EXT.shape[1]), bounds=[(0, C) for _ in LTR], factr=1.0)
    w_hat = (vrow(alphaStar) * vrow(ZTR) * DTR_EXT).sum(1)
    w, b = w_hat[0:DTR.shape[0]], w_hat[-1] * K
    return w, b

# GMM
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

# Risk Functions
def compute_confusion_matrix(predictedLabels, classLabels):
    nClasses = classLabels.max() + 1
    M = np.zeros((nClasses, nClasses), dtype=np.int32)
    for i in range(classLabels.size):
        M[predictedLabels[i], classLabels[i]] += 1
    return M

def compute_empirical_Bayes_risk_binary(predictedLabels, classLabels, prior, Cfn, Cfp, normalize=True):
    M = compute_confusion_matrix(predictedLabels, classLabels)
    Pfn = M[0,1] / (M[0,1] + M[1,1])
    Pfp = M[1,0] / (M[0,0] + M[1,0])
    bayesError = prior * Cfn * Pfn + (1-prior) * Cfp * Pfp
    if normalize:
        return bayesError / np.minimum(prior * Cfn, (1-prior)*Cfp)
    return bayesError

def compute_optimal_Bayes_binary_llr(llr, prior, Cfn, Cfp):
    th = -np.log( (prior * Cfn) / ((1 - prior) * Cfp) )
    return np.int32(llr > th)

def compute_empirical_Bayes_risk_binary_llr_optimal_decisions(llr, classLabels, prior, Cfn, Cfp, normalize=True):
    predictedLabels = compute_optimal_Bayes_binary_llr(llr, prior, Cfn, Cfp)
    return compute_empirical_Bayes_risk_binary(predictedLabels, classLabels, prior, Cfn, Cfp, normalize=normalize)

def compute_minDCF_binary_fast(llr, classLabels, prior, Cfn, Cfp, returnThreshold=False):
    Pfn, Pfp, th = compute_Pfn_Pfp_allThresholds_fast(llr, classLabels)
    minDCF = (prior * Cfn * Pfn + (1 - prior) * Cfp * Pfp) / np.minimum(prior * Cfn, (1-prior)*Cfp)
    idx = np.argmin(minDCF)
    if returnThreshold:
        return minDCF[idx], th[idx]
    else:
        return minDCF[idx]

def compute_Pfn_Pfp_allThresholds_fast(llr, classLabels):
    llrSorter = np.argsort(llr)
    llrSorted = llr[llrSorter]
    classLabelsSorted = classLabels[llrSorter]
    Pfp = []
    Pfn = []
    nTrue = (classLabelsSorted==1).sum()
    nFalse = (classLabelsSorted==0).sum()
    nFalseNegative = 0
    nFalsePositive = nFalse
    Pfn.append(nFalseNegative / nTrue)
    Pfp.append(nFalsePositive / nFalse)
    for idx in range(len(llrSorted)):
        if classLabelsSorted[idx] == 1:
            nFalseNegative += 1
        if classLabelsSorted[idx] == 0:
            nFalsePositive -= 1
        Pfn.append(nFalseNegative / nTrue)
        Pfp.append(nFalsePositive / nFalse)
    llrSorted = np.concatenate([-np.array([np.inf]), llrSorted])
    PfnOut = []
    PfpOut = []
    thresholdsOut = []
    for idx in range(len(llrSorted)):
        if idx == len(llrSorted) - 1 or llrSorted[idx+1] != llrSorted[idx]:
            PfnOut.append(Pfn[idx])
            PfpOut.append(Pfp[idx])
            thresholdsOut.append(llrSorted[idx])
    return np.array(PfnOut), np.array(PfpOut), np.array(thresholdsOut)

compute_actDCF_binary_fast = compute_empirical_Bayes_risk_binary_llr_optimal_decisions

# Load data
D, L = load_iris_binary()

# Split data
(DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)

# Define priors and lambdas
priors = [0.1, 0.5, 0.9]
lambdas = [1e-3, 1e-1, 1.0]

# Logistic Regression Results
logreg_results = {}

for piT in priors:
    logreg_results[piT] = []
    for l in lambdas:
        w, b = trainLogRegBinary(DTR, LTR, l)
        scores = np.dot(w.T, DVAL) + b
        minDCF = compute_minDCF_binary_fast(scores, LVAL, piT, 1.0, 1.0)
        actDCF = compute_actDCF_binary_fast(scores, LVAL, piT, 1.0, 1.0)
        logreg_results[piT].append((l, minDCF, actDCF))
        print(f'LogReg - Prior: {piT}, Lambda: {l}, minDCF: {minDCF:.4f}, actDCF: {actDCF:.4f}')

# SVM Results
Cs = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
svm_results = {}

for piT in priors:
    svm_results[piT] = []
    for C in Cs:
        w, b = train_dual_SVM_linear(DTR, LTR, C)
        scores = (np.dot(w.T, DVAL) + b).ravel()
        minDCF = compute_minDCF_binary_fast(scores, LVAL, piT, 1.0, 1.0)
        actDCF = compute_actDCF_binary_fast(scores, LVAL, piT, 1.0, 1.0)
        svm_results[piT].append((C, minDCF, actDCF))
        print(f'SVM - Prior: {piT}, C: {C}, minDCF: {minDCF:.4f}, actDCF: {actDCF:.4f}')

# GMM Results
components = [1, 2, 4, 8, 16]
gmm_results = {}

for piT in priors:
    gmm_results[piT] = []
    for num_components in components:
        gmm0 = train_GMM_LBG_EM(DTR[:, LTR == 0], num_components, verbose=False)
        gmm1 = train_GMM_LBG_EM(DTR[:, LTR == 1], num_components, verbose=False)
        scores = logpdf_GMM(DVAL, gmm1) - logpdf_GMM(DVAL, gmm0)
        minDCF = compute_minDCF_binary_fast(scores, LVAL, piT, 1.0, 1.0)
        actDCF = compute_actDCF_binary_fast(scores, LVAL, piT, 1.0, 1.0)
        gmm_results[piT].append((num_components, minDCF, actDCF))
        print(f'GMM - Prior: {piT}, Components: {num_components}, minDCF: {minDCF:.4f}, actDCF: {actDCF:.4f}')

# Plot results
def plot_results(results, param_name, title):
    for piT, result in results.items():
        params, minDCFs, actDCFs = zip(*result)
        plt.plot(params, minDCFs, marker='o', label=f'Prior {piT} - minDCF')
        plt.plot(params, actDCFs, marker='x', label=f'Prior {piT} - actDCF')

    plt.xscale('log')
    plt.xlabel(param_name)
    plt.ylabel('DCF')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

plot_results(logreg_results, 'Lambda', 'DCF vs. Lambda for Logistic Regression')
plot_results(svm_results, 'C', 'DCF vs. C for SVM')
plot_results(gmm_results, 'Components', 'DCF vs. Number of Components for GMM')
