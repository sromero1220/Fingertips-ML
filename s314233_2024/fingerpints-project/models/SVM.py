import numpy as np
import pandas as pd
import math
import os
from tabulate import tabulate
import sys
import matplotlib.pyplot as plt
from scipy.optimize import fmin_l_bfgs_b

sys.path.append(os.path.abspath("MLandPattern"))
import MLandPattern as ML

def load(filepath):
    with open(filepath) as fp:
        content = fp.readlines()
    num_features = content[0].split(',').__len__() - 1

    num_samples = 0
    labels = []
    features = []
    for line in content:
        num_samples += 1
        feature = []
        fields = line.split(',')
        for i in range(num_features):
            feature.append(float(fields[i]))
        features.append(np.array(feature).reshape(num_features, 1))
        labels.append(int(fields[num_features].split('\n')[0]))

    mat = np.hstack(features)
    np.random.seed(0)
    idx = np.random.permutation(num_samples)

    return (mat[:, idx], np.array(labels)[idx])

################################################ Cross Validations###########################################################################


def crossValidation_SVM(K, DTR, LTR, kappa, C, pi, pit, allscoresCondition=False):
    minDCFvalues1 = []
    minDCFvalues3 = []
    minDCFvalues5 = []
    N = DTR.shape[1]
    seglen = N//K
    alls = []
    for c in C:
        alls = []
        DTE_cap = []
        for i in range(K):
            mask = [True] * N
            mask[i*seglen: (i+1)*seglen] = [False] * seglen
            notmask = [not x for x in mask]
            DTRr = DTR[:, mask]
            LTRr = LTR[mask]
            DTV = DTR[:, notmask]

            w_cap = SVM_2C(DTRr, LTRr, kappa, c, pit)
            nsamples = DTV.shape[1]
            nattr = DTV.shape[0] + 1
            DTE_cap = np.append(
                DTV.ravel(), [kappa] * nsamples).reshape(nattr, nsamples)
            scores = np.dot(w_cap, DTE_cap)
            alls += scores.tolist()
        if allscoresCondition:
            return alls
        else:
            for p in pi:
                minDCF = minimum_detection_cost(p, 1, 1, alls, LTR)
                if p == 0.1:
                    minDCFvalues1.append(minDCF)
                    print("minDCF 0.1: ", minDCF, "C=", c)
                if p == 0.3:
                    minDCFvalues3.append(minDCF)
                    print("minDCF 0.3: ", minDCF, "C=", c)
                if p == 0.5:
                    minDCFvalues5.append(minDCF)
                    print("minDCF 0.5: ", minDCF, "C=", c)
    return minDCFvalues1, minDCFvalues3, minDCFvalues5


def crossValidation_SVM_poly(K, DTR, LTR, kappa, C, degree, c, pi, pit, allscoresCondition=False):
    minDCFvalues1 = []
    minDCFvalues3 = []
    minDCFvalues5 = []
    N = DTR.shape[1]
    seglen = N//K
    alls = []
    for constrain in C:
        alls = []
        for i in range(K):
            mask = [True] * N
            mask[i*seglen: (i+1)*seglen] = [False] * seglen
            notmask = [not x for x in mask]
            DTRr = DTR[:, mask]
            LTRr = LTR[mask]
            DTV = DTR[:, notmask]

            a, z, feat = SVM_2C_ndegree(
                DTRr, LTRr, kappa, constrain, degree, c, pit)
            scores = []
            for i in range(DTV.shape[1]):
                scores.append(classify_SVM_2C_ndegree(
                    DTV[:, i], a, z, feat, degree, kappa, c))
            alls += scores
        if allscoresCondition:
            return alls
        else:
            for p in pi:
                minDCF = minimum_detection_cost(p, 1, 1, alls, LTR)
                if p == 0.1:
                    minDCFvalues1.append(minDCF)
                    print("minDCF 0.1: ", minDCF, "C=", constrain)
                if p == 0.3:
                    minDCFvalues3.append(minDCF)
                    print("minDCF 0.3: ", minDCF, "C=", constrain)
                if p == 0.5:
                    minDCFvalues5.append(minDCF)
                    print("minDCF 0.5: ", minDCF, "C=", constrain)

    return minDCFvalues1, minDCFvalues3, minDCFvalues5


def crossValidation_SVM_RBF(K, DTR, LTR, kappa, constrain, gamma, pi, pit, allscoresCondition=False):
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

        a, zi, feat = SVM_RBF_wrapper(DTRr, LTRr, kappa, constrain, gamma, pit)
        scores = []
        for i in range(DTV.shape[1]):
            scores.append(classify_SVM_RBF(
                DTV[:, i], a, zi, feat, gamma, kappa))

        allscores += scores
    minDCF = minimum_detection_cost(pi, 1, 1, allscores, LTR)
    print("minDCF: ", minDCF, "C=", constrain)
    if allscoresCondition:
        return allscores
    return minDCF

########################################### Functions cross validations##################################################################


def SVM_2C(feat, lab, K, C, pit):
    x0 = np.zeros(feat.shape[1])
    bnds = np.array([(0, C)] * feat.shape[1])
    TL = np.array(lab)
    TL = TL.reshape(-1, 1)
    pitemp = 400/2371
    pitempNTL = 1-pitemp
    pitempTL = pitemp
    pitNTL = 1-pit
    pitTL = pit
    bnds = np.where(TL == 0, bnds*(pitNTL/pitempNTL), bnds*(pitTL/pitempTL))

    nsamples = feat.shape[1]
    nattr = feat.shape[0] + 1
    DTR_cap = np.append(feat.ravel(), [K] * nsamples).reshape(nattr, nsamples)
    z = np.array([1 if a == 1 else -1 for a in lab]).reshape(1, nsamples)
    SVMobj = SVMClass(feat, lab, K, DTR_cap, z)

    alpha, _1, _2 = fmin_l_bfgs_b(
        SVMobj.SVM_func, x0, SVMobj.SVM_grad, bounds=bnds, factr=1.0)

    w_cap = np.sum(alpha.reshape(1, nsamples) * z * DTR_cap, axis=1)

    return w_cap


def SVM_2C_ndegree(feat, lab, K, C, degree, c, pit):
    x0 = np.zeros(feat.shape[1])
    bnds = np.array([(0, C)] * feat.shape[1])
    TL = np.array(lab)
    TL = TL.reshape(-1, 1)
    pitemp = 400/2371
    pitempNTL = 1-pitemp
    pitempTL = pitemp
    pitNTL = 1-pit
    pitTL = pit
    bnds = np.where(TL == 0, bnds*(pitNTL/pitempNTL), bnds*(pitTL/pitempTL))

    nsamples = feat.shape[1]
    z = np.array([1 if a == 1 else -1 for a in lab]).reshape(1, nsamples)
    SVMobj = SVMClass_ndegree(feat, K, z, degree, c)

    alpha, _1, _2 = fmin_l_bfgs_b(
        SVMobj.SVM_func_ndegree, x0, SVMobj.SVM_grad_ndegree, bounds=bnds, factr=1.0)

    return alpha, z, feat


def classify_SVM_2C_ndegree(sample, alpha, z, feat, degree, K, c):
    chi = K**2
    score = (alpha * z * ((np.dot(feat.T, sample) + c) ** degree + chi)).sum()
    return score


def SVM_RBF_wrapper(feat, lab, K, C, gamma, pit):
    x0 = np.zeros(feat.shape[1])
    bnds = np.array([(0, C)] * feat.shape[1])
    TL = np.array(lab)
    TL = TL.reshape(-1, 1)
    pitemp = 400/2371
    pitempNTL = 1-pitemp
    pitempTL = pitemp
    pitNTL = 1-pit
    pitTL = pit
    bnds = np.where(TL == 0, bnds*(pitNTL/pitempNTL), bnds*(pitTL/pitempTL))

    nsamples = feat.shape[1]
    z = np.array([1 if a == 1 else -1 for a in lab]).reshape(1, nsamples)
    SVMobj = SVM_RBF(feat, K, z, gamma)

    alpha, _1, _2 = fmin_l_bfgs_b(
        SVMobj.SVM_func_ndegree, x0, SVMobj.SVM_grad_ndegree, bounds=bnds, factr=1.0)

    return alpha, z, feat


def classify_SVM_RBF(sample, alpha, z, feat, gamma, K):
    score = 0
    for i in range(feat.shape[1]):
        tmp = (alpha[i]*z[:, i]*(math.exp(-gamma *
               np.dot((sample-feat[:, i]).T, sample-feat[:, i]))+K**2))[0]
        score = score+tmp
    return score  # >0


########################################### Classes cross validation###################################################################
class SVMClass:
    def __init__(self, DTR, LTR, K, D_cap, z):
        self.LTR = LTR
        self.K = K
        self.n = LTR.shape[0]
        self.z = z
        self.D_cap = D_cap
        G = np.dot(self.D_cap.T, self.D_cap)
        self.H_cap = np.dot(self.z.T, self.z) * G

    def SVM_func(self, a):
        alpha = a.reshape(a.size, 1)
        return (0.5 * np.dot(np.dot(alpha.T, self.H_cap), alpha) - alpha.sum())

    def SVM_grad(self, a):
        alpha = a.reshape(a.size, 1)
        return (np.dot(self.H_cap, alpha) - 1).ravel()

    def duality_gap(self, a, w_cap, C):
        w = w_cap.reshape(w_cap.size, 1)
        arr = (1 - (self.z * np.dot(w.T, self.D_cap))).ravel()
        print("primal loss: ", (0.5 * np.dot(w.T, w) + C *
              (np.array([0 if el < 0 else el for el in arr]).sum())))
        print("dual loss: ", self.SVM_func(a))
        return self.SVM_func(a) + (0.5 * np.dot(w.T, w) + C * (np.array([0 if el < 0 else el for el in arr]).sum()))


class SVMClass_ndegree:
    def __init__(self, DTR, K, z, degree, c):
        chi = K**2
        self.H_cap = np.dot(z.T, z) * \
            (((np.dot(DTR.T, DTR) + c) ** degree) + chi)

    def SVM_func_ndegree(self, a):
        alpha = a.reshape(a.size, 1)
        return (0.5 * np.dot(np.dot(alpha.T, self.H_cap), alpha) - alpha.sum())

    def SVM_grad_ndegree(self, a):
        alpha = a.reshape(a.size, 1)
        return (np.dot(self.H_cap, alpha) - 1).ravel()


class SVM_RBF:
    def __init__(self, DTR, K, z, gamma):
        chi = K**2
        l = []
        for i in range(DTR.shape[1]):
            for j in range(DTR.shape[1]):
                tmp = -gamma * \
                    np.dot((DTR[:, i]-DTR[:, j]).T, DTR[:, i]-DTR[:, j])
                l.append(tmp)
        l = np.array(l).reshape(DTR.shape[1], DTR.shape[1])
        self.H_cap = (np.exp(l)+chi)*np.dot(z.T, z)

    def SVM_func_ndegree(self, a):
        alpha = a.reshape(a.size, 1)
        return (0.5 * np.dot(np.dot(alpha.T, self.H_cap), alpha) - alpha.sum())

    def SVM_grad_ndegree(self, a):
        alpha = a.reshape(a.size, 1)
        return (np.dot(self.H_cap, alpha) - 1).ravel()

 ########################################## MinDCF##############################################################


def minimum_detection_cost(pi, cfn, cfp, scores, labels):
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
        minDCF = (DCFu / DCFdummy) if (DCFu / DCFdummy) < minDCF else minDCF

    return minDCF


def confusion_matrix(predictions, labels, nclasses):
    CM = np.array([0] * nclasses**2).reshape(nclasses, nclasses)
    for i in range(predictions.size):
        CM[predictions[i], round(labels[i])] += 1
    return CM


###############################################################################################
if __name__ == "__main__":
    path = os.path.abspath("data/Train.txt")
    [full_train_att, full_train_label] = load(path)

    pit = 0.5
    pi = [0.1, 0.3, 0.5]
    Cfn = 1
    Cfp = 1
    initial_C = np.logspace(-5, 5, 15)
    initial_K = 1
    k = 5

    kappa = initial_K * np.power(10, 0)
    ### --------------K-fold----------------------###

    minDCFvalues1F = []
    minDCFvalues3F = []
    minDCFvalues5F = []

    minDCFvalues1F, minDCFvalues3F, minDCFvalues5F = crossValidation_SVM(
        k, full_train_att, full_train_label, kappa, initial_C, pi, pit)

    ### --------z-score----------###
    standard_deviation = np.std(full_train_att)
    z_data = ML.center_data(full_train_att) / standard_deviation

    minDCFvalues1Z = []
    minDCFvalues3Z = []
    minDCFvalues5Z = []

    minDCFvalues1Z, minDCFvalues3Z, minDCFvalues5Z = crossValidation_SVM(
        k, z_data, full_train_label, kappa, initial_C, pi, pit)

    ## -----------------Polynomial---------###

    initial_d = 2

    minDCFvalues1Q = []
    minDCFvalues3Q = []
    minDCFvalues5Q = []

    minDCFvalues1Q, minDCFvalues3Q, minDCFvalues5Q = crossValidation_SVM_poly(
        k, full_train_att, full_train_label, kappa, initial_C, initial_d, 1, pi, pit)

    ### ----------------Pol-z----------------###
    minDCFvalues1Qz = []
    minDCFvalues3Qz = []
    minDCFvalues5Qz = []

    minDCFvalues1Qz, minDCFvalues3Qz, minDCFvalues5Qz = crossValidation_SVM_poly(
        k, z_data, full_train_label, kappa, initial_C, initial_d, 1, pi, pit)

    # -----------------------Radial--------------------###

    gamma = [10**-1, 10**-2, 10**-3]
    pi = 0.5
    minDCFvalues1R = []
    minDCFvalues2R = []
    minDCFvalues3R = []
    for g in gamma:
        for C in initial_C:
            minDCF = crossValidation_SVM_RBF(
                k, full_train_att, full_train_label, kappa, C, g, pi, pit)
            if g == 10**-1:
                minDCFvalues1R.append(minDCF)
            if g == 10**-2:
                minDCFvalues2R.append(minDCF)
            if g == 10**-3:
                minDCFvalues3R.append(minDCF)
    ## -----------------------Radial--------------------###

    minDCFvalues1Rz = []
    minDCFvalues2Rz = []
    minDCFvalues3Rz = []
    for g in gamma:
        for C in initial_C:
            minDCF = crossValidation_SVM_RBF(
                k, z_data, full_train_label, kappa, C, g, pi, pit)
            if g == 10**-1:
                minDCFvalues1Rz.append(minDCF)
            if g == 10**-2:
                minDCFvalues2Rz.append(minDCF)
            if g == 10**-3:
                minDCFvalues3Rz.append(minDCF)


######################################## Graphs##################################################
    print("SVM with k-fold={k}")
    plt.figure(figsize=(10, 6))
    plt.semilogx(initial_C, minDCFvalues1F, label=f'π= 0.1')
    plt.semilogx(initial_C, minDCFvalues3F, label=f'π= 0.3')
    plt.semilogx(initial_C, minDCFvalues5F, label=f'π= 0.5')
    plt.xlabel('C')
    plt.ylabel('minDCF')
    plt.title('Full SVM regression')
    plt.legend()
    plt.show()

    print("SVM with Z-norm")
    plt.figure(figsize=(10, 6))
    plt.semilogx(initial_C, minDCFvalues1Z, label=f'π= 0.1')
    plt.semilogx(initial_C, minDCFvalues3Z, label=f'π= 0.3')
    plt.semilogx(initial_C, minDCFvalues5Z, label=f'π= 0.5')
    plt.xlabel('C')
    plt.ylabel('minDCF')
    plt.title('Z-Norm SVM regression')
    plt.legend()
    plt.show()

    print("Polynomial kernel degree 2")
    plt.figure(figsize=(10, 6))
    plt.semilogx(initial_C, minDCFvalues1Q, label=f'π= 0.1')
    plt.semilogx(initial_C, minDCFvalues3Q, label=f'π= 0.3')
    plt.semilogx(initial_C, minDCFvalues5Q, label=f'π= 0.5')
    plt.xlabel('C')
    plt.ylabel('minDCF')
    plt.title('Polynomial kernel degree 2')
    plt.legend()
    plt.show()

    print("Polynomial kernel degree 2 Z-norm")
    plt.figure(figsize=(10, 6))
    plt.semilogx(initial_C, minDCFvalues1Qz, label=f'π= 0.1')
    plt.semilogx(initial_C, minDCFvalues3Qz, label=f'π= 0.3')
    plt.semilogx(initial_C, minDCFvalues5Qz, label=f'π= 0.5')
    plt.xlabel('C')
    plt.ylabel('minDCF')
    plt.title('Polynomial kernel degree 2 Z-norm')
    plt.legend()
    plt.show()

    print("Radial SVM")
    plt.figure(figsize=(10, 6))
    plt.semilogx(initial_C, minDCFvalues1R, label=f'log(gamma)= -1')
    plt.semilogx(initial_C, minDCFvalues2R, label=f'log(gamma)= -2')
    plt.semilogx(initial_C, minDCFvalues3R, label=f'log(gamma)= -3')
    plt.xlabel('C')
    plt.ylabel('minDCF')
    plt.title('Full Radial SVM')
    plt.legend()
    plt.show()

    print("Radial SVM z-norm")
    plt.figure(figsize=(10, 6))
    plt.semilogx(initial_C, minDCFvalues1Rz, label=f'log(gamma)= -1')
    plt.semilogx(initial_C, minDCFvalues2Rz, label=f'log(gamma)= -2')
    plt.semilogx(initial_C, minDCFvalues3Rz, label=f'log(gamma)= -3')
    plt.xlabel('C')
    plt.ylabel('minDCF')
    plt.title('Full Radial SVM Z-norm')
    plt.legend()
    plt.show()

 ############################################# Tables####################################################
    searchC = float(input("Enter a C value you want to search for in SVM: "))
    searchC = [searchC]
    searchQC = float(input("Enter a C value you want to search for in QSVM: "))
    searchQC = [searchQC]
    searchG = float(
        input("Enter a gamma value you want to search for in RBSVM: "))
    searchRC = float(
        input("Enter a C value you want to search for in RBSVM: "))
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
    cont = 0
    for i in q:
        for x in pit:
            if i == 0:
                tableKFold.append([f"SVM pit={x}"])
                minDCFvalues1T, minDCFvalues3T, minDCFvalues5T = crossValidation_SVM(
                    k, full_train_att, full_train_label, kappa, searchC, pi, x)
                tableKFold[cont].extend(
                    [minDCFvalues1T, minDCFvalues3T, minDCFvalues5T])
                minDCFvalues1Tz, minDCFvalues3Tz, minDCFvalues5Tz = crossValidation_SVM(
                    k, z_data, full_train_label, kappa, searchC, pi, x)
                tableKFold[cont].extend(
                    [minDCFvalues1Tz, minDCFvalues3Tz, minDCFvalues5Tz])
                cont += 1
            if i == 1:
                tableKFold.append([f"QSVM pit={x}"])
                minDCFvalues1TQ, minDCFvalues3TQ, minDCFvalues5TQ = crossValidation_SVM_poly(
                    k, full_train_att, full_train_label, kappa, searchQC, initial_d, 1, pi, x)
                tableKFold[cont].extend(
                    [minDCFvalues1TQ, minDCFvalues3TQ, minDCFvalues5TQ])
                minDCFvalues1TQz, minDCFvalues3TQz, minDCFvalues5TQz = crossValidation_SVM_poly(
                    k, z_data, full_train_label, kappa, searchQC, initial_d, 1, pi, x)
                tableKFold[cont].extend(
                    [minDCFvalues1TQz, minDCFvalues3TQz, minDCFvalues5TQz])
                cont += 1
            if i == 2:
                temp = [f"RBSVM pit= {x}"]
                for p in pi:
                    minDCF = crossValidation_SVM_RBF(
                        k, full_train_att, full_train_label, kappa, searchRC, searchG, p, x)
                    temp.append(minDCF)
                for p in pi:
                    minDCF = crossValidation_SVM_RBF(
                        k, z_data, full_train_label, kappa, searchRC, searchG, p, x)
                    temp.append(minDCF)
                tableKFold.append(temp)
                cont += 1

    print(tabulate(tableKFold, headers))
