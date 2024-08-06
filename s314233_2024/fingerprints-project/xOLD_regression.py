import numpy as np
import pandas as pd
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


def vrow(arr):
    return arr.reshape(1, arr.shape[0])


def vcol(arr):
    if (arr.shape.__len__() == 1):
        return arr.reshape(arr.shape[0], 1)
    return arr.reshape(arr.shape[1], 1)


def crossValidation_logreg(K, DTR, LTR, l_list, nclasses, model, pi, pit, allscoresCondition=False):
    minDCFvalues1 = []
    minDCFvalues3 = []
    minDCFvalues5 = []
    N = DTR.shape[1]
    seglen = N//K

    for l in l_list:
        alls = []
        for i in range(K):
            mask = [True] * N
            mask[i*seglen: (i+1)*seglen] = [False] * seglen
            notmask = [not x for x in mask]
            DTRr = DTR[:, mask]
            LTRr = LTR[mask]
            DTV = DTR[:, notmask]
            w, b = model(DTRr, LTRr, l, pit)
            scores = np.dot(vrow(w), DTV) + b
            alls += scores.ravel().tolist()
        if allscoresCondition:
            return alls
        else:
            for p in pi:
                minDCF = minimum_detection_cost(p, 1, 1, alls, LTR)
                print("minDCF: ", minDCF)
                if p == 0.1:
                    minDCFvalues1.append(minDCF)
                if p == 0.3:
                    minDCFvalues3.append(minDCF)
                if p == 0.5:
                    minDCFvalues5.append(minDCF)
    return minDCFvalues1, minDCFvalues3, minDCFvalues5


def linear_regression(feat, lab, l, pit):
    x0 = np.zeros(feat.shape[0] + 1)
    logRegObj = logRegClass(feat, lab, l, pit)

    opt, f, d = fmin_l_bfgs_b(logRegObj.logreg_obj, x0, approx_grad=True)
    w, b = opt[0:-1], opt[-1]

    return w, b


class logRegClass:
    def __init__(self, DTR, LTR, l, pit):
        self.DTR = DTR
        self.LTR = LTR
        self.l = l
        self.pit = pit
        self.n = LTR.shape[0]

    def logreg_obj(self, v):
        nt = np.count_nonzero(self.LTR == 1)
        nf = np.count_nonzero(self.LTR != 1)
        w, b = v[0:-1], v[-1]
        zi = self.LTR * 2 - 1

        male = self.DTR[:, self.LTR == 0]
        female = self.DTR[:, self.LTR == 1]

        log_sumt = 0
        log_sumf = 0
        zif = zi[zi == -1]
        zit = zi[zi == 1]

        inter_sol = -zit * (np.dot(w.T, female) + b)
        log_sumt = np.sum(np.logaddexp(0, inter_sol))

        inter_sol = -zif * (np.dot(w.T, male) + b)
        log_sumf = np.sum(np.logaddexp(0, inter_sol))
        retFunc = self.l / 2 * \
            np.power(np.linalg.norm(w), 2) + self.pit / nt * \
            log_sumt + (1-self.pit)/nf * log_sumf

        return retFunc


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


def expande_feature_space(data):
    rt = []
    for i in range(data.shape[1]):
        rt.append(expanded_feature_space(vcol(data[:, i])))
    return np.hstack(rt)


def expanded_feature_space(sample):
    return vcol(np.concatenate((np.dot(sample, sample.T).ravel(), sample.ravel())))


def crossValidation_logreg_onepoint(K, DTR, LTR, l, nclasses, model, pi, pit, allscoresCondition=False):
    N = DTR.shape[1]
    seglen = N//K
    alls = []
    for i in range(K):
        mask = [True] * N
        mask[i*seglen: (i+1)*seglen] = [False] * seglen
        notmask = [not x for x in mask]
        DTRr = DTR[:, mask]
        LTRr = LTR[mask]
        DTV = DTR[:, notmask]
        w, b = model(DTRr, LTRr, l, pit)
        scores = np.dot(vrow(w), DTV) + b
        alls += scores.ravel().tolist()
        minDCF = minimum_detection_cost(pi, 1, 1, alls, LTR)
    print("minDCF: ", minDCF)
    if allscoresCondition:
        return alls
    return minDCF


if __name__ == "__main__":
    tableKFold = []
    pi = [0.1, 0.3, 0.5]
    pit = 0.5
    Cfn = 1
    Cfp = 1
    l_list = np.logspace(-4, 2, 13)
    path = os.path.abspath("Train.txt")
    [full_train_att, full_train_label] = load(path)
    q = [0, 1]
    k = 5

    feat_expanded = expande_feature_space(full_train_att)
    for i in q:
        if i == 1:
            minDCFvalues1, minDCFvalues3, minDCFvalues5 = crossValidation_logreg(
                k, feat_expanded, full_train_label, l_list, 2, linear_regression, pi, pit)
        if i == 0:
            minDCFvalues1, minDCFvalues3, minDCFvalues5 = crossValidation_logreg(
                k, full_train_att, full_train_label, l_list, 2, linear_regression, pi, pit)
        if i == 1:
            print(f"Quadratic regression with k-fold = {k}")
        if i == 0:
            print(f"Logarithmic regression with k-fold = {k}")
        plt.figure(figsize=(10, 6))
        plt.semilogx(l_list, minDCFvalues1, label=f'π= 0.1')
        plt.semilogx(l_list, minDCFvalues3, label=f'π= 0.3')
        plt.semilogx(l_list, minDCFvalues5, label=f'π= 0.5')
        plt.xlabel('lambda')
        plt.ylabel('minDCF')
        if i == 0:
            plt.title('Logarithmic regression')
        if i == 1:
            plt.title('Quadratic regression')
        plt.legend()
        plt.show()

    # Z-norm
    standard_deviation = np.std(full_train_att)
    z_data = ML.center_data(full_train_att) / standard_deviation
    feat_expanded_z = expande_feature_space(z_data)
    for i in q:
        if i == 1:
            minDCFvalues1, minDCFvalues3, minDCFvalues5 = crossValidation_logreg(
                k, feat_expanded_z, full_train_label, l_list, 2, linear_regression, pi, pit)
        if i == 0:
            minDCFvalues1, minDCFvalues3, minDCFvalues5 = crossValidation_logreg(
                k, z_data, full_train_label, l_list, 2, linear_regression, pi, pit)
        if i == 1:
            print(f"Quadratic regression with k-fold = {k} z-norm")
        if i == 0:
            print(f"Logarithmic regression with k-fold = {k} z-norm")
        plt.figure(figsize=(10, 6))
        plt.semilogx(l_list, minDCFvalues1, label=f'π= 0.1')
        plt.semilogx(l_list, minDCFvalues3, label=f'π= 0.3')
        plt.semilogx(l_list, minDCFvalues5, label=f'π= 0.5')
        plt.xlabel('lambda')
        plt.ylabel('minDCF')
        if i == 0:
            plt.title('Logarithmic regression Z-norm')
        if i == 1:
            plt.title('Quadratic regression Z-norm')
        plt.legend()
        plt.show()

    pt, _ = ML.PCA(full_train_att, 5)
    att_PCA5 = np.dot(pt, full_train_att)
    feat_expanded_PCA = expande_feature_space(att_PCA5)
    for i in q:
        if i == 1:
            minDCFvalues1, minDCFvalues3, minDCFvalues5 = crossValidation_logreg(
                k, feat_expanded_PCA, full_train_label, l_list, 2, linear_regression, pi, pit)
        if i == 0:
            minDCFvalues1, minDCFvalues3, minDCFvalues5 = crossValidation_logreg(
                k, att_PCA5, full_train_label, l_list, 2, linear_regression, pi, pit)
        if i == 1:
            print(f"Quadratic regression with k-fold = {k} PCA= 5")
        if i == 0:
            print(f"Logarithmic regression with k-fold = {k} PCA= 5")
        plt.figure(figsize=(10, 6))
        plt.semilogx(l_list, minDCFvalues1, label=f'π= 0.1')
        plt.semilogx(l_list, minDCFvalues3, label=f'π= 0.3')
        plt.semilogx(l_list, minDCFvalues5, label=f'π= 0.5')
        plt.xlabel('lambda')
        plt.ylabel('minDCF')
        if i == 0:
            plt.title('Logarithmic regression PCA 5')
        if i == 1:
            plt.title('Quadratic regression PCA 5')
        plt.legend()
        plt.show()

    pt, _ = ML.PCA(z_data, 5)
    att_PCA5_z = np.dot(pt, z_data)
    feat_expanded_PCA_z = expande_feature_space(att_PCA5)
    for i in q:
        if i == 1:
            minDCFvalues1, minDCFvalues3, minDCFvalues5 = crossValidation_logreg(
                k, feat_expanded_PCA_z, full_train_label, l_list, 2, linear_regression, pi, pit)
        if i == 0:
            minDCFvalues1, minDCFvalues3, minDCFvalues5 = crossValidation_logreg(
                k, att_PCA5_z, full_train_label, l_list, 2, linear_regression, pi, pit)
        if i == 1:
            print(f"Quadratic regression with k-fold = {k} z-norm PCA= 5")
        if i == 0:
            print(f"Logarithmic regression with k-fold = {k} z-norm PCA= 5")
        plt.figure(figsize=(10, 6))
        plt.semilogx(l_list, minDCFvalues1, label=f'π= 0.1')
        plt.semilogx(l_list, minDCFvalues3, label=f'π= 0.3')
        plt.semilogx(l_list, minDCFvalues5, label=f'π= 0.5')
        plt.xlabel('lambda')
        plt.ylabel('minDCF')
        if i == 0:
            plt.title('Logarithmic regression z-norm PCA 5')
        if i == 1:
            plt.title('Quadratic regression z-norm PCA 5')
        plt.legend()
        plt.show()

    searchL = float(input("Enter a lambda value you want to search for: "))
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
    cont = 0
    for i in q:
        for x in pit:
            if i == 0:
                tableKFold.append([f"LR {x}"])
                for p in pi:
                    minDCF = crossValidation_logreg_onepoint(
                        k, full_train_att, full_train_label, searchL, 2, linear_regression, p, x)
                    tableKFold[cont].append(minDCF)
                for p in pi:
                    minDCF = crossValidation_logreg_onepoint(
                        k, z_data, full_train_label, searchL, 2, linear_regression, p, x)
                    tableKFold[cont].append(minDCF)
            if i == 1:
                tableKFold.append([f"QLR {x}"])
                for p in pi:
                    minDCF = crossValidation_logreg_onepoint(
                        k, feat_expanded, full_train_label, searchL, 2, linear_regression, p, x)
                    tableKFold[cont].append(minDCF)
                for p in pi:
                    minDCF = crossValidation_logreg_onepoint(
                        k, feat_expanded_z, full_train_label, searchL, 2, linear_regression, p, x)
                    tableKFold[cont].append(minDCF)

            cont += 1

    print(tabulate(tableKFold, headers))
