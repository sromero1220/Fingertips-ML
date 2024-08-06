from SVM import crossValidation_SVM_RBF, crossValidation_SVM_poly, SVM_RBF_wrapper, classify_SVM_RBF, SVM_2C_ndegree, classify_SVM_2C_ndegree
from regression import crossValidation_logreg, linear_regression
from GMM import crossValidation_GMM, classify_GMM, GMM_classifier_tied
import os
import sys
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import fmin_l_bfgs_b
sys.path.append(os.path.abspath("MLandPattern"))
import MLandPattern as ML


def load_data(filepath):
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


def expande_feature_space(data):
    rt = []
    for i in range(data.shape[1]):
        rt.append(expanded_feature_space(vcol(data[:, i])))
    return np.hstack(rt)


def expanded_feature_space(sample):
    return vcol(np.concatenate((np.dot(sample, sample.T).ravel(), sample.ravel())))


def bayes_error_plot(llratios, labels, color):
    effPriorLogOdds = np.linspace(-3, 3, 21)
    pis = [1 / (1 + np.exp(- p)) for p in effPriorLogOdds]

    dcf = [compute_DCF_binary(pi, 1, 1, llratios, labels) for pi in pis]
    mindcf = [minDCFcalc(pi, 1, 1, llratios, labels) for pi in pis]

    plt.plot(effPriorLogOdds, dcf, label='DCF', color=color)
    plt.plot(effPriorLogOdds, mindcf, linestyle='dashed',
             label='min DCF', color=color)
    plt.ylim([0, 1.1])
    plt.xlim([-3, 3])


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
        minDCF = (DCFu / DCFdummy) if (DCFu / DCFdummy) < minDCF else minDCF

    return minDCF


def confusion_matrix(predictions, labels, nclasses):
    CM = np.array([0] * nclasses**2).reshape(nclasses, nclasses)
    for i in range(predictions.size):
        CM[predictions[i], round(labels[i])] += 1
    return CM


def compute_DCF_binary(pi, cfn, cfp, llratios, labels):
    CM, _ = optimal_bayes_decisions_binary(pi, cfn, cfp, llratios, labels)
    fnr = CM[0, 1] / (CM[1, 1] + CM[0, 1])
    fpr = CM[1, 0] / (CM[0, 0] + CM[1, 0])
    DCFu = pi*cfn*fnr + (1-pi)*cfp*fpr
    DCFdummy = min(pi*cfn, (1-pi)*cfp)
    return DCFu / DCFdummy


def optimal_bayes_decisions_binary(pi, cfn, cfp, llratios, labels):
    t = -np.log(pi * cfn / ((1 - pi) * cfp))
    # print(llratios.shape)
    predictions = [1 if el > t else 0 for el in llratios]
    return (confusion_matrix(np.array(predictions), labels, 2), t)


def crossValidation_logreg_cal(K, DTR, LTR, l):
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
        wopt, bopt = linear_regression_cal(DTRr, LTRr, l)
        scores = np.dot(vrow(wopt), DTV) + bopt
        alls += scores.ravel().tolist()
    print("minDCF: ", minDCFcalc(0.5, 1, 1, alls, LTR))
    return alls


def linear_regression_cal(feat, lab, l):
    x0 = np.zeros(feat.shape[0] + 1)
    logRegObj = logRegClass(feat, lab, l)

    opt, _1, _2 = fmin_l_bfgs_b(logRegObj.logreg_obj, x0, approx_grad=True)
    wopt, bopt = opt[0:-1], opt[-1]

    return wopt, bopt


class logRegClass:
    def __init__(self, DTR, LTR, l):
        self.DTR = DTR
        self.LTR = LTR
        self.l = l
        self.n = LTR.shape[0]

    def logreg_obj(self, v):
        w, b = v[0:-1], v[-1]
        J = self.l/2 * np.dot(w, w.T)
        for i in range(self.n):
            zi = 1 if self.LTR[i] == 1 else -1
            J += np.logaddexp(0, -zi *
                              (np.dot(w.T, self.DTR[:, i]) + b)) / self.n

        return J


def DET(llratios, labels, color='b', show=True):
    thresholds = np.array(llratios)
    thresholds.sort()
    # fnr instead of tpr

    fnrs = []
    fprs = []
    for t in thresholds:
        predictions = [1 if el > t else 0 for el in llratios]
        CM = confusion_matrix(np.array(predictions), labels, 2)
        fnrs.append(CM[0, 1] / (CM[0, 1] + CM[1, 1]))
        fprs.append(CM[1, 0] / (CM[0, 0] + CM[1, 0]))

    plt.plot(fprs, fnrs, color=color)
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel("FPR")
    plt.ylabel("FNR")
    plt.grid(True)
    if (show):
        plt.show()


def ROC(llratios, labels, color='b', show=True):
    thresholds = np.array(llratios)
    thresholds.sort()

    tprs = []
    fprs = []
    for t in thresholds:
        predictions = [1 if el > t else 0 for el in llratios]
        CM = confusion_matrix(np.array(predictions), labels, 2)
        tprs.append(1 - (CM[0, 1] / (CM[1, 1] + CM[0, 1])))
        fprs.append(CM[1, 0] / (CM[0, 0] + CM[1, 0]))

    plt.plot(fprs, tprs, color=color)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.grid(True)
    if (show):
        plt.show()


###########################################################################################
if __name__ == "__main__":
    path = os.path.abspath("data/Train.txt")
    full_train_att, full_train_label = load_data(path)
    path = os.path.abspath("data/Test.txt")
    full_test_att, full_test_label = load_data(path)

    K = 5
    steps = 2

    standard_deviation = np.std(full_train_att)
    z_data = ML.center_data(full_train_att) / standard_deviation
    feat_expanded = expande_feature_space(full_train_att)
    feat_expanded_z = expande_feature_space(z_data)

    standard_deviation_test = np.std(full_test_att)
    full_test_z = ML.center_data(full_test_att) / standard_deviation_test
    feat_tested_expand = expande_feature_space(full_test_att)
    feat_expanded_test_z = expande_feature_space(full_test_z)

    initial_K = 1
    kappa = initial_K * np.power(10, 0)

    model1_1 = crossValidation_SVM_RBF(
        K, z_data, full_train_label, kappa, 100, 0.01, 0.1, 0.5, True)
    model1_5 = crossValidation_SVM_RBF(
        K, z_data, full_train_label, kappa, 100, 0.01, 0.5, 0.5, True)
    model1_avg = (np.array(model1_1) + np.array(model1_5)) / 2

    model2_1 = crossValidation_SVM_poly(K, z_data, full_train_label, kappa, [
                                        0.1], 2, 1, [0.1], 0.5, True)
    model2_5 = crossValidation_SVM_poly(K, z_data, full_train_label, kappa, [
                                        0.1], 2, 1, [0.5], 0.5, True)
    model2_avg = (np.array(model2_1) + np.array(model2_5)) / 2

    model3_1 = crossValidation_logreg(K, feat_expanded_z, full_train_label, [
                                      0.01], 2, linear_regression, [0.1], 0.5, True)
    model3_5 = crossValidation_logreg(K, feat_expanded_z, full_train_label, [
                                      0.01], 2, linear_regression, [0.5], 0.5, True)
    model3_avg = (np.array(model3_1) + np.array(model3_5)) / 2

    model4_1 = crossValidation_GMM(
        K, full_train_att, full_train_label, "GMM tied", 32, 0.1, 0.000001, 0.01, 0.1, True)
    model4_5 = crossValidation_GMM(
        K, full_train_att, full_train_label, "GMM tied", 32, 0.1, 0.000001, 0.01, 0.5, True)
    model4_avg = (np.array(model4_1) + np.array(model4_5)) / 2

    bayes_error_plot(model1_avg, full_train_label, 'r')
    plt.title('RBF SVM uncalibrated')
    plt.legend(['actDCF', 'minDCF'])
    plt.show()
    bayes_error_plot(model2_avg, full_train_label, 'b')
    plt.title('Polynomial SVM uncalibrated')
    plt.legend(['actDCF', 'minDCF'])
    plt.show()
    bayes_error_plot(model3_avg, full_train_label, 'g')
    plt.title('Quadratic Regression uncalibrated')
    plt.legend(['actDCF', 'minDCF'])
    plt.show()
    bayes_error_plot(model4_avg, full_train_label, 'y')
    plt.title('Tied GMM uncalibrated')
    plt.legend(['actDCF', 'minDCF'])
    plt.show()

    np.random.seed(1)
    idx = np.random.permutation(len(model1_avg))
    model1_avg = vrow(np.array(model1_avg)[idx])
    model2_avg = vrow(np.array(model2_avg)[idx])
    model3_avg = vrow(np.array(model3_avg)[idx])
    model4_avg = vrow(np.array(model4_avg)[idx])
    lab_shuffle = full_train_label[idx]

    el = -5
    sc1_cal = crossValidation_logreg(K, model1_avg, lab_shuffle, [10**el], 2, linear_regression, [0.1], 0.5, True)
    sc2_cal = crossValidation_logreg(K, model2_avg, lab_shuffle, [10**el], 2, linear_regression, [0.1], 0.5, True)
    sc3_cal = crossValidation_logreg(K, model3_avg, lab_shuffle, [10**el], 2, linear_regression, [0.1], 0.5, True)
    sc4_cal = crossValidation_logreg(K, model4_avg, lab_shuffle, [10**el], 2, linear_regression, [0.1], 0.5, True)

    bayes_error_plot(sc1_cal, lab_shuffle, 'r')
    plt.title('RBF SVM calibrated')
    plt.legend(['actDCF', 'minDCF'])
    plt.show()

    bayes_error_plot(sc2_cal, lab_shuffle, 'b')
    plt.title('Polynomial SVM calibrated')
    plt.legend(['actDCF', 'minDCF'])
    plt.show()

    bayes_error_plot(sc3_cal, lab_shuffle, 'g')
    plt.title('Quadratic Regression calibrated')
    plt.legend(['actDCF', 'minDCF'])
    plt.show()

    bayes_error_plot(sc4_cal, lab_shuffle, 'y')
    plt.title('Tied GMM calibrated')
    plt.legend(['actDCF', 'minDCF'])
    plt.show()

    print('Evaluation')
    a, zi, _ = SVM_RBF_wrapper(z_data, full_train_label, kappa, 100, 0.01, 0.5)
    scores_RBSVM = []
    for i in range(full_test_z.shape[1]):
        scores_RBSVM.append(classify_SVM_RBF(
            full_test_z[:, i], a, full_test_z, z_data, 0.001, kappa))

    a, z, _ = SVM_2C_ndegree(z_data, full_train_label, kappa, 0.1, 2, 1, 0.5)
    scores_QSVM = []
    for i in range(full_test_att.shape[1]):
        scores_QSVM.append(classify_SVM_2C_ndegree(
            full_test_z[:, i], a, z, z_data, 2, kappa, 1))

    w, b = linear_regression(feat_expanded_z, full_train_label, 0.01, 0.5)
    scores_QLR = (np.dot(vrow(w), feat_expanded_test_z) + b)[0]

    gmms_tied = GMM_classifier_tied(
        full_train_att, full_train_label, 32, 0.1, 0.000001, 0.01)
    scores_GMM_tied = classify_GMM(full_test_att, full_test_label, gmms_tied)

    el = -5
    sc1 = vrow(np.array(model1_avg)[0])
    sc2 = vrow(np.array(model2_avg)[0])
    sc3 = vrow(np.array(model3_avg)[0])
    sc4 = vrow(np.array(model4_avg)[0])

    scores_RBSVM = vrow(np.array(scores_RBSVM))
    scores_QSVM = vrow(np.array(scores_QSVM))
    scores_QLR = vrow(np.array(scores_QLR))
    scores_GMM_tied = vrow(np.array(scores_GMM_tied))

    wopt_rbf, bopt_rbf = linear_regression(sc1, lab_shuffle, 10**el, 0.5)
    wopt_qsvm, bopt_qsvm = linear_regression(sc2, lab_shuffle, 10**el, 0.5)
    wopt_qlr, bopt_qlr = linear_regression(sc3, lab_shuffle, 10**el, 0.5)
    wopt_gmm, bopt_gmm = linear_regression(sc4, lab_shuffle, 10**el, 0.5)

    predictions_rbf = (np.dot(vrow(wopt_rbf), scores_RBSVM) + bopt_rbf)[0]
    predictions_qsvm = (np.dot(vrow(wopt_qsvm), scores_QSVM) + bopt_qsvm)[0]
    predictions_qlr = (np.dot(vrow(wopt_qlr), scores_QLR) + bopt_qlr)[0]
    predictions_gmm = (np.dot(vrow(wopt_gmm), scores_GMM_tied) + bopt_gmm)[0]

    print('final model accuracy:', ((predictions_rbf > 0)
          == full_test_label).sum() / len(predictions_rbf))

    bayes_error_plot(predictions_rbf, full_test_label, 'r')
    plt.title('RBF SVM')
    plt.legend(['act', 'min'])
    plt.show()

    bayes_error_plot(predictions_qsvm, full_test_label, 'b')
    plt.title('Polynomial SVM')
    plt.legend(['act', 'min'])
    plt.show()

    bayes_error_plot(predictions_qlr, full_test_label, 'g')
    plt.title('Quadratic Regression')
    plt.legend(['act', 'min'])
    plt.show()

    bayes_error_plot(predictions_gmm, full_test_label, 'y')
    plt.title('Tied GMM')
    plt.legend(['act', 'min'])
    plt.show()

    minDCF1 = minDCFcalc(
        0.1, 1, 1, predictions_rbf, full_test_label)
    minDCF5 = minDCFcalc(
        0.5, 1, 1, predictions_rbf, full_test_label)
    minDCFavg = (minDCF1 + minDCF5) / 2
    print('calibrated RBF SVM minDCFAvg: ', minDCFavg)
    print('calibrated RBF SVM minDCF1: ', minDCF1)
    print('calibrated RBF SVM minDCF5: ', minDCF5)

    minDCF1 = minDCFcalc(
        0.1, 1, 1, predictions_qsvm, full_test_label)
    minDCF5 = minDCFcalc(
        0.5, 1, 1, predictions_qsvm, full_test_label)
    minDCFavg = (minDCF1 + minDCF5) / 2
    print('calibrated Polynomial SVM minDCFAvg: ', minDCFavg)
    print('calibrated Polynomial SVM minDCF1: ', minDCF1)
    print('calibrated Polynomial SVM minDCF5: ', minDCF5)

    minDCF1 = minDCFcalc(
        0.1, 1, 1, predictions_qlr, full_test_label)
    minDCF5 = minDCFcalc(
        0.5, 1, 1, predictions_qlr, full_test_label)
    minDCFavg = (minDCF1 + minDCF5) / 2
    print('calibrated Quadratic Regression minDCFAvg: ', minDCFavg)
    print('calibrated Quadratic Regression minDCF1: ', minDCF1)
    print('calibrated Quadratic Regression minDCF5: ', minDCF5)

    minDCF1 = minDCFcalc(
        0.1, 1, 1, predictions_gmm, full_test_label)
    minDCF5 = minDCFcalc(
        0.5, 1, 1, predictions_gmm, full_test_label)
    minDCFavg = (minDCF1 + minDCF5) / 2
    print('calibrated Tied GMM minDCFAvg: ', minDCFavg)
    print('calibrated Tied GMM minDCF1: ', minDCF1)
    print('calibrated Tied GMM minDCF5: ', minDCF5)

    DET(predictions_rbf, full_test_label, color='r', show=False)
    DET(predictions_qsvm, full_test_label, color='b', show=False)
    DET(predictions_qlr, full_test_label, color='g', show=False)
    DET(predictions_gmm, full_test_label, color='y', show=False)
    plt.legend(['RBF', 'QSVM', 'QLR', 'GMM'])
    plt.show()

    ROC(predictions_rbf, full_test_label, color='r', show=False)
    ROC(predictions_qsvm, full_test_label, color='b', show=False)
    ROC(predictions_qlr, full_test_label, color='g', show=False)
    ROC(predictions_gmm, full_test_label, color='y', show=False)
    plt.legend(['RBF', 'QSVM', 'QLR', 'GMM'])
    plt.show()
