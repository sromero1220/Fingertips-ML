import numpy as np
import pandas as pd
import scipy
import os
import sys
from tabulate import tabulate
from matplotlib import pyplot as plt
sys.path.append(os.path.abspath("MLandPattern"))
import MLandPattern as ML

def load(filepath):
    with open(filepath) as fp:
        content = fp.readlines()
    nfeatures = content[0].split(',').__len__() - 1

    nsamples = 0
    labels = []
    features = []
    for line in content:
        nsamples += 1
        feature = []
        fields = line.split(',')
        for i in range(nfeatures):
            feature.append(float(fields[i]))
        features.append(np.array(feature).reshape(nfeatures, 1))
        labels.append(int(fields[nfeatures].split('\n')[0]))
    mat = np.hstack(features)
    np.random.seed(0)
    idx = np.random.permutation(nsamples)

    return (mat[:, idx], np.array(labels)[idx])


################################### cross validation########################################################
def crossValidation_GMM(K, DTR, LTR, model, ng, alpha, threshold, psi, pi=0.5, allscoresCondition=False):
    N = DTR.shape[1]
    seglen = N//K
    allscores = []
    for i in range(K):
        mask = [True] * N
        mask[i*seglen: (i+1)*seglen] = [False] * seglen
        notmask = [not x for x in mask]
        DTRr = DTR[:, mask]
        LTRr = LTR[mask]
        DTV = DTR[:, notmask]
        LTV = LTR[notmask]
        if model == "GMM full":
            gmms = GMM_classifier_full(DTRr, LTRr, ng, alpha, threshold, psi)
        elif model == "GMM diagonal":
            gmms = GMM_classifier_diagonal(
                DTRr, LTRr, ng, alpha, threshold, psi)
        elif model == "GMM tied":
            gmms = GMM_classifier_tied(DTRr, LTRr, ng, alpha, threshold, psi)
        llr = classify_GMM(DTV, LTV, gmms)
        allscores += llr.tolist()
    minDCF = minimum_detection_cost(pi, 1, 1, allscores, LTR)
    print("minDCF: ", minDCF)
    if allscoresCondition:
        return allscores
    return minDCF

############################ Functions#########################


def dsmean(mat):
    return vcol(mat.mean(1))


def dscovmatrix(mat):
    DC = mat - dsmean(mat)
    return np.dot(DC, DC.T) / mat.shape[1]


def vcol(arr):
    if (arr.shape.__len__() == 1):
        return arr.reshape(arr.shape[0], 1)
    return arr.reshape(arr.shape[1], 1)


def classify_GMM(DTE, LTE, gmms):
    scores = []
    for g in gmms:
        scores.append(logpdf_GMM(DTE, g))
    scores = np.vstack(scores)
    return np.log(scores[0, :] / scores[1, :])


def logpdf_GMM(X, gmm):
    M = len(gmm)
    S = []
    for i in range(M):
        S.append(logpdf_GAU_ND(X, gmm[i][1], gmm[i][2]) + np.log(gmm[i][0]))
    S = np.vstack(S)
    logdens = scipy.special.logsumexp(S, axis=0)
    return logdens


def logpdf_GAU_ND(x, mu, C):
    M = x.shape[0]
    _, logs = np.linalg.slogdet(C)
    siginv = np.linalg.inv(C)
    rt = []
    for i in range(x.shape[1]):
        sample = vcol(x[:, i])
        a = -M/2*np.log(2*np.pi) - 1/2*logs - 1/2 * \
            np.dot(np.dot((sample-mu).T, siginv), (sample-mu))
        rt.append(a)
    return np.array(rt).ravel()


############################ Models#########################
def GMM_classifier_full(DTR, LTR, ng, alpha, threshold, psi):
    nclasses = len(set(LTR))
    gmms = []
    for c in range(nclasses):
        data = DTR[:, LTR == c]
        gmms.append(LBG_ng_constrained(data, ng, alpha, threshold, psi))
    return gmms


def LBG_ng_constrained(X, ng, alpha, threshold, psi, gmstart=False):
    if (gmstart == False):
        cov = dscovmatrix(X)
        U, s, _ = np.linalg.svd(cov)
        s[s < psi] = psi
        cov = np.dot(U, vcol(s)*U.T)
        gmm_start = [(1.0, dsmean(X), cov)]
    else:
        gmm_start = gmstart
    ncomp = 1
    while ncomp < ng:
        gmm_new = []
        for i in range(len(gmm_start)):
            U, s, Vh = np.linalg.svd(gmm_start[i][2])
            d = U[:, 0:1] * s[0]**0.5 * alpha
            gmm_new.append(
                (gmm_start[i][0]/2, gmm_start[i][1] + d, gmm_start[i][2]))
            gmm_new.append(
                (gmm_start[i][0]/2, gmm_start[i][1] - d, gmm_start[i][2]))
        gmm_start = GMM_EM_constrained(X, gmm_new, threshold, psi)
        ncomp *= 2
    return gmm_start


def GMM_EM_constrained(X, gmm, threshold, psi):
    ll_curr = 0
    N = X.shape[1]
    D = X.shape[0]
    ll_prev = logpdf_GMM(X, gmm).sum()/N - 2*threshold
    M = len(gmm)

    while 1:
        # E step
        S = []
        for i in range(M):
            S.append(logpdf_GAU_ND(X, gmm[i][1],
                     gmm[i][2]) + np.log(gmm[i][0]))
        S = np.vstack(S)
        marginals = scipy.special.logsumexp(S, axis=0)
        posteriors = np.exp(S - marginals)
        ll_curr = marginals.sum() / N
        if (ll_curr < ll_prev):
            return gmm
        if ((ll_curr - ll_prev) < threshold):
            return gmm
        ll_prev = ll_curr

        gmm_n = []
        Zgsum = 0
        for i in range(M):
            Zg = posteriors[i, :].sum()
            Zgsum += Zg
            Fg = (posteriors[i, :] * X).sum(axis=1).reshape(D, 1)
            Sg = np.array([0] * D ** 2).reshape(D, D)
            for j in range(N):
                s = X[:, j].reshape(D, 1)
                Sg = Sg + (posteriors[i, j] * np.dot(s, s.T))
            mu = Fg/Zg
            sigma = Sg/Zg - np.dot(mu, mu.T)
            U, s, _ = np.linalg.svd(sigma)
            s[s < psi] = psi
            sigma = np.dot(U, vcol(s)*U.T)
            gmm_n.append([Zg, mu, sigma])
        for i in range(M):
            gmm_n[i][0] /= Zgsum
        gmm = gmm_n


def GMM_classifier_diagonal(DTR, LTR, ng, alpha, threshold, psi):
    nclasses = len(set(LTR))
    gmms = []
    for c in range(nclasses):
        data = DTR[:, LTR == c]
        gmms.append(LBG_ng_constrained_diagonal(
            data, ng, alpha, threshold, psi))
    return gmms


def LBG_ng_constrained_diagonal(X, ng, alpha, threshold, psi, gmstart=False):
    if (gmstart == False):
        cov = dscovmatrix(X)
        U, s, _ = np.linalg.svd(cov)
        s[s < psi] = psi
        cov = np.eye(X.shape[0]) * np.dot(U, vcol(s)*U.T)
        gmm_start = [(1.0, dsmean(X), cov)]
    else:
        gmm_start = gmstart
    ncomp = 1
    while ncomp < ng:
        gmm_new = []
        for i in range(len(gmm_start)):
            U, s, Vh = np.linalg.svd(gmm_start[i][2])
            d = U[:, 0:1] * s[0]**0.5 * alpha
            gmm_new.append(
                (gmm_start[i][0]/2, gmm_start[i][1] + d, gmm_start[i][2]))
            gmm_new.append(
                (gmm_start[i][0]/2, gmm_start[i][1] - d, gmm_start[i][2]))
        gmm_start = GMM_EM_constrained_diagonal(X, gmm_new, threshold, psi)
        ncomp *= 2
    return gmm_start


def GMM_EM_constrained_diagonal(X, gmm, threshold, psi):
    ll_curr = 0
    N = X.shape[1]
    D = X.shape[0]
    ll_prev = logpdf_GMM(X, gmm).sum()/N - 2*threshold
    M = len(gmm)

    while 1:
        # E step
        S = []
        for i in range(M):
            S.append(logpdf_GAU_ND(X, gmm[i][1],
                     gmm[i][2]) + np.log(gmm[i][0]))
        S = np.vstack(S)
        marginals = scipy.special.logsumexp(S, axis=0)
        posteriors = np.exp(S - marginals)
        ll_curr = marginals.sum() / N
        if (ll_curr < ll_prev):
            return gmm
        if ((ll_curr - ll_prev) < threshold):
            return gmm
        ll_prev = ll_curr

        gmm_n = []
        Zgsum = 0
        for i in range(M):
            Zg = posteriors[i, :].sum()
            Zgsum += Zg
            Fg = (posteriors[i, :] * X).sum(axis=1).reshape(D, 1)
            Sg = np.array([0] * D ** 2).reshape(D, D)
            for j in range(N):
                s = X[:, j].reshape(D, 1)
                Sg = Sg + (posteriors[i, j] * np.dot(s, s.T))
            mu = Fg/Zg
            sigma = np.eye(D) * (Sg/Zg - np.dot(mu, mu.T))
            U, s, _ = np.linalg.svd(sigma)
            s[s < psi] = psi
            sigma = np.dot(U, vcol(s)*U.T)
            gmm_n.append([Zg, mu, sigma])
        for i in range(M):
            gmm_n[i][0] /= Zgsum
        gmm = gmm_n


def GMM_classifier_tied(DTR, LTR, ng, alpha, threshold, psi):
    nclasses = len(set(LTR))
    gmms = []
    for c in range(nclasses):
        data = DTR[:, LTR == c]
        gmms.append(LBG_ng_constrained_tied(data, ng, alpha, threshold, psi))
    return gmms


def LBG_ng_constrained_tied(X, ng, alpha, threshold, psi, gmstart=False):
    if (gmstart == False):
        cov = dscovmatrix(X)
        U, s, _ = np.linalg.svd(cov)
        s[s < psi] = psi
        cov = np.dot(U, vcol(s)*U.T)
        gmm_start = [(1.0, dsmean(X), cov)]
    else:
        gmm_start = gmstart
    ncomp = 1
    while ncomp < ng:
        gmm_new = []
        for i in range(len(gmm_start)):
            U, s, Vh = np.linalg.svd(gmm_start[i][2])
            d = U[:, 0:1] * s[0]**0.5 * alpha
            gmm_new.append(
                (gmm_start[i][0]/2, gmm_start[i][1] + d, gmm_start[i][2]))
            gmm_new.append(
                (gmm_start[i][0]/2, gmm_start[i][1] - d, gmm_start[i][2]))
        gmm_start = GMM_EM_constrained_tied(X, gmm_new, threshold, psi)
        ncomp *= 2
    return gmm_start


def GMM_EM_constrained_tied(X, gmm, threshold, psi):

    ll_curr = 0
    N = X.shape[1]
    D = X.shape[0]
    ll_prev = logpdf_GMM(X, gmm).sum()/N - 2*threshold
    M = len(gmm)

    while 1:
        # E step
        S = []
        for i in range(M):
            S.append(logpdf_GAU_ND(X, gmm[i][1],
                     gmm[i][2]) + np.log(gmm[i][0]))
        S = np.vstack(S)
        marginals = scipy.special.logsumexp(S, axis=0)
        posteriors = np.exp(S - marginals)
        ll_curr = marginals.sum() / N
        if (ll_curr < ll_prev):
            return gmm
        if ((ll_curr - ll_prev) < threshold):
            return gmm
        ll_prev = ll_curr

        # M step
        gmm_n = []
        Zgsum = 0
        Sgsum = np.array([0] * D ** 2).reshape(D, D)
        for i in range(M):
            Zg = posteriors[i, :].sum()
            Zgsum += Zg
            Fg = (posteriors[i, :] * X).sum(axis=1).reshape(D, 1)
            Sg = np.array([0] * D ** 2).reshape(D, D)
            for j in range(N):
                s = X[:, j].reshape(D, 1)
                Sg = Sg + (posteriors[i, j] * np.dot(s, s.T))
            mu = Fg/Zg
            sigma = Sg/Zg - np.dot(mu, mu.T)
            U, s, _ = np.linalg.svd(sigma)
            s[s < psi] = psi
            Sgsum = Sgsum + ((Zg/N) * np.dot(U, vcol(s)*U.T))
            gmm_n.append([Zg, mu, sigma])
        for i in range(M):
            gmm_n[i][0] /= Zgsum
            gmm_n[i][2] = Sgsum
        gmm = gmm_n


############################################## minDCF##########################################
def minimum_detection_cost(pi, cfn, cfp, scores, labels):
    thresholds = np.append(np.min(scores) - 1, scores)
    thresholds = np.append(thresholds, np.max(scores) + 1)
    thresholds.sort()
    DCFdummy = min(pi*cfn, (1-pi)*cfp)
    minDCF = 100000
    for t in thresholds:
        predictions = np.array(scores > t, dtype=int)
        CM = confusion_matrix(predictions, labels, 2)
        fnr = CM[0, 1] / CM[:, 1].sum()
        fpr = CM[1, 0] / CM[:, 0].sum()
        DCFu = pi*cfn*fnr + (1-pi)*cfp*fpr
        minDCF = min(DCFu / DCFdummy, minDCF)

    return minDCF


def confusion_matrix(predictions, labels, nclasses):
    CM = np.array([0] * nclasses**2).reshape(nclasses, nclasses)
    for i in range(predictions.size):
        CM[predictions[i], round(labels[i])] += 1
    return CM
###########################################################################################


def call_GMM(attributes, labels, models, k_value=5, pi=[0.5]):
    total_iter = len(models)*len(pi)
    perc = 0
    result_minDCF = []
    c2 = 1
    for model in models:
        ref = model.split(":")
        mod = ref[0]
        constrains = eval(ref[1])
        alpha = constrains[0]
        niter = constrains[1]
        psi = constrains[2]
        for p in pi:
            minDCF = crossValidation_GMM(
                k_value, attributes, labels, mod, niter, alpha, 0.000001, psi, p)
            perc += 1
            print(f"{round(perc * 100 / total_iter, 2)}%")
        result_minDCF.append(minDCF)
        c2 += 1

    return result_minDCF


def graph_data(raw_results, z_results, model, pca=0):
    attribute = {
        "Raw": raw_results,
        "Z-Score": z_results
    }

    maxVal = max(max(attribute["Raw"]), max(attribute["Z-Score"]))

    x = np.arange(len(raw_results))
    print(x)
    width = 0.25
    multiplier = 0

    labels = np.power(2, x)

    fig, ax = plt.subplots(layout='constrained')

    for attribute, measurement in attribute.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        ax.bar_label(rects, padding=3)
        multiplier += 1

    ax.set_ylabel('minDCF')
    ax.set_xticks(x + width/2, labels)
    ax.legend(loc='upper left', ncols=3)
    ax.set_ylim(0, maxVal + 0.1)
    if pca != 0:
        plt.savefig(f"{os.getcwd()}/Image/GMM-{model}-PCA.png")
    else:
        plt.savefig(f"{os.getcwd()}/Image/GMM-{model}.png")
    plt.show()


if __name__ == "__main__":

    path = "data\Train.txt"
    [full_train_att, full_train_label] = load(path)
    pi = [0.5]
    standard_deviation = np.std(full_train_att)
    z_data = ML.center_data(full_train_att) / standard_deviation

    pt, _ = ML.PCA(full_train_att, 5)
    att_PCA5 = np.dot(pt, full_train_att)
    pt, _ = ML.PCA(z_data, 5)
    att_PCA5_z = np.dot(pt, z_data)

    # GMM FUll
    model = "GMM full"
    headers = [f"{model}:[0.1, 1, 0.01]", f"{model}:[0.1, 2, 0.01]", f"{model}:[0.1, 4, 0.01]",
               f"{model}:[0.1, 8, 0.01]", f"{model}:[0.1, 16, 0.01]", f"{model}:[0.1, 32, 0.01]"]
    print("Full data")
    raw_values = call_GMM(
        full_train_att, full_train_label, pi=pi, models=headers)
    print("Z-norm data")
    z_values = call_GMM(z_data, full_train_label, pi=pi, models=headers)

    # GMM Diagonal
    model = "GMM diagonal"
    headers = [f"{model}:[0.1, 1, 0.01]", f"{model}:[0.1, 2, 0.01]", f"{model}:[0.1, 4, 0.01]",
               f"{model}:[0.1, 8, 0.01]", f"{model}:[0.1, 16, 0.01]", f"{model}:[0.1, 32, 0.01]"]
    print("Full data GMM diagonal")
    raw_values_d = call_GMM(
        full_train_att, full_train_label, pi=pi, models=headers)
    print("Z-norm data GMM diagonal")
    z_values_d = call_GMM(z_data, full_train_label, pi=pi, models=headers)

    # GMM Tied
    model = "GMM tied"
    headers = [f"{model}:[0.1, 1, 0.01]", f"{model}:[0.1, 2, 0.01]", f"{model}:[0.1, 4, 0.01]",
               f"{model}:[0.1, 8, 0.01]", f"{model}:[0.1, 16, 0.01]", f"{model}:[0.1, 32, 0.01]"]
    print("Full data GMM tied")
    raw_values_t = call_GMM(
        full_train_att, full_train_label, pi=pi, models=headers)
    print("Z-norm data GMM tied")
    z_values_t = call_GMM(z_data, full_train_label, pi=pi, models=headers)

    # GMM FUll PCA 5
    model = "GMM full"
    headers = [f"{model}:[0.1, 1, 0.01]", f"{model}:[0.1, 2, 0.01]", f"{model}:[0.1, 4, 0.01]",
               f"{model}:[0.1, 8, 0.01]", f"{model}:[0.1, 16, 0.01]", f"{model}:[0.1, 32, 0.01]"]
    print("Full data PCA 5")
    raw_values_pca = call_GMM(
        att_PCA5, full_train_label, pi=pi, models=headers)
    print("Z-norm data PCA 5")
    z_values_pca = call_GMM(
        att_PCA5_z, full_train_label, pi=pi, models=headers)

    # GMM Diagonal PCA 5
    model = "GMM diagonal"
    headers = [f"{model}:[0.1, 1, 0.01]", f"{model}:[0.1, 2, 0.01]", f"{model}:[0.1, 4, 0.01]",
               f"{model}:[0.1, 8, 0.01]", f"{model}:[0.1, 16, 0.01]", f"{model}:[0.1, 32, 0.01]"]
    print("Full data GMM diagonal PCA 5")
    raw_values_d_pca = call_GMM(
        att_PCA5, full_train_label, pi=pi, models=headers)
    print("Z-norm data GMM diagonal PCA 5")
    z_values_d_pca = call_GMM(
        att_PCA5_z, full_train_label, pi=pi, models=headers)

    # GMM Tied PCA 5
    model = "GMM tied"
    headers = [f"{model}:[0.1, 1, 0.01]", f"{model}:[0.1, 2, 0.01]", f"{model}:[0.1, 4, 0.01]",
               f"{model}:[0.1, 8, 0.01]", f"{model}:[0.1, 16, 0.01]", f"{model}:[0.1, 32, 0.01]"]
    print("Full data GMM tied PCA 5")
    raw_values_t_pca = call_GMM(
        att_PCA5, full_train_label, pi=pi, models=headers)
    print("Z-norm data GMM tied PCA 5")
    z_values_t_pca = call_GMM(
        att_PCA5_z, full_train_label, pi=pi, models=headers)

    graph_data(raw_values, z_values, "GMM", pca=0)
    graph_data(raw_values_d, z_values_d, "GMM_Diagonal", pca=0)
    graph_data(raw_values_t, z_values_t, "GMM_Tied", pca=0)
    graph_data(raw_values_pca, z_values_pca, "GMM_PCA5", pca=5)
    graph_data(raw_values_d_pca, z_values_d_pca, "GMM_Diagonal_PCA5", pca=5)
    graph_data(raw_values_t_pca, z_values_t_pca, "GMM_Tied_PCA5", pca=5)

    ########################### Table################################
    headers = [
        "Model",
        "π=0.1 minDCF",
        "π=0.3 minDCF",
        "π=0.5 minDCF",
        "π=0.1 Z-norm minDCF",
        "π=0.3 Z-norm minDCF",
        "π=0.5 Z-norm minDCF",
    ]
    pit = [0.1, 0.3, 0.5]
    pi = [0.1, 0.3, 0.5]
    q = [0, 1, 2]
    tableKFold = []
    tablePCA = []
    cont = 0
    k_value = 5
    standard_deviation = np.std(full_train_att)
    z_data = ML.center_data(full_train_att) / standard_deviation
    pt, _ = ML.PCA(full_train_att, 5)
    att_PCA5 = np.dot(pt, full_train_att)
    pt, _ = ML.PCA(z_data, 5)
    att_PCA5_z = np.dot(pt, z_data)

    tableKFold.append(["GMM full 2"])
    cont = 0
    for p in pi:
        minDCF = crossValidation_GMM(
            k_value, full_train_att, full_train_label, "GMM full", 2, 0.1, 0.000001, 0.01, p)
        tableKFold[cont].extend([minDCF])
    for p in pi:
        minDCF = crossValidation_GMM(
            k_value, z_data, full_train_label, "GMM full", 2, 0.1, 0.000001, 0.01, p)
        tableKFold[cont].extend([minDCF])
    cont += 1
    tableKFold.append(["GMM diagonal 16"])
    for p in pi:
        minDCF = crossValidation_GMM(
            k_value, full_train_att, full_train_label, "GMM diagonal", 16, 0.1, 0.000001, 0.01, p)
        tableKFold[cont].extend([minDCF])
    for p in pi:
        minDCF = crossValidation_GMM(
            k_value, z_data, full_train_label, "GMM diagonal", 16, 0.1, 0.000001, 0.01, p)
        tableKFold[cont].extend([minDCF])
    cont += 1
    tableKFold.append(["GMM diagonal 4"])
    for p in pi:
        minDCF = crossValidation_GMM(
            k_value, full_train_att, full_train_label, "GMM diagonal", 4, 0.1, 0.000001, 0.01, p)
        tableKFold[cont].extend([minDCF])
    for p in pi:
        minDCF = crossValidation_GMM(
            k_value, z_data, full_train_label, "GMM diagonal", 4, 0.1, 0.000001, 0.01, p)
        tableKFold[cont].extend([minDCF])
    cont += 1
    tableKFold.append(["GMM tied 32"])
    for p in pi:
        minDCF = crossValidation_GMM(
            k_value, full_train_att, full_train_label, "GMM tied", 32, 0.1, 0.000001, 0.01, p)
        tableKFold[cont].extend([minDCF])
    for p in pi:
        minDCF = crossValidation_GMM(
            k_value, z_data, full_train_label, "GMM tied", 32, 0.1, 0.000001, 0.01, p)
        tableKFold[cont].extend([minDCF])
    cont += 1
    print(tabulate(tableKFold, headers))

    cont = 0
    tablePCA.append(["GMM PCA5 full 8"])
    for p in pi:
        minDCF = crossValidation_GMM(
            k_value, att_PCA5, full_train_label, "GMM full", 8, 0.1, 0.000001, 0.01, p)
        tablePCA[cont].extend([minDCF])
    for p in pi:
        minDCF = crossValidation_GMM(
            k_value, att_PCA5_z, full_train_label, "GMM full", 8, 0.1, 0.000001, 0.01, p)
        tablePCA[cont].extend([minDCF])
    cont += 1
    tablePCA.append(["GMM PCA5 diagonal 2"])
    for p in pi:
        minDCF = crossValidation_GMM(
            k_value, att_PCA5, full_train_label, "GMM diagonal", 2, 0.1, 0.000001, 0.01, p)
        tablePCA[cont].extend([minDCF])
    for p in pi:
        minDCF = crossValidation_GMM(
            k_value, att_PCA5_z, full_train_label, "GMM diagonal", 2, 0.1, 0.000001, 0.01, p)
        tablePCA[cont].extend([minDCF])
    cont += 1
    tablePCA.append(["GMM PCA5 tied 16"])
    for p in pi:
        minDCF = crossValidation_GMM(
            k_value, att_PCA5, full_train_label, "GMM tied", 16, 0.1, 0.000001, 0.01, p)
        tablePCA[cont].extend([minDCF])
    for p in pi:
        minDCF = crossValidation_GMM(
            k_value, att_PCA5_z, full_train_label, "GMM tied", 16, 0.1, 0.000001, 0.01, p)
        tablePCA[cont].extend([minDCF])
    cont += 1
    tablePCA.append(["GMM PCA5 tied 4"])
    for p in pi:
        minDCF = crossValidation_GMM(
            k_value, att_PCA5, full_train_label, "GMM tied", 4, 0.1, 0.000001, 0.01, p)
        tablePCA[cont].extend([minDCF])
    for p in pi:
        minDCF = crossValidation_GMM(
            k_value, att_PCA5_z, full_train_label, "GMM tied", 4, 0.1, 0.000001, 0.01, p)
        tablePCA[cont].extend([minDCF])
    cont += 1
    print(tabulate(tablePCA, headers))
