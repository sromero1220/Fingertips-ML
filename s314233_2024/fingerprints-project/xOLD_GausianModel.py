import os
import sys
import numpy as np
sys.path.append(os.path.abspath("MLandPattern"))
import MLandPattern as ml

def load_data(filepath):
    with open(filepath) as file:
        content = file.readlines()
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


def vectorize_column(arr):
    if (arr.shape.__len__() == 1):
        return arr.reshape(arr.shape[0], 1)
    return arr.reshape(arr.shape[1], 1)


def vectorize_row(arr):
    return arr.reshape(1, arr.shape[0])


def mean_of_data_set(mat):
    return vectorize_column(mat.mean(1))


def covariance_matrix(mat):
    deviation_from_mean = mat - mean_of_data_set(mat)
    return np.dot(deviation_from_mean, deviation_from_mean.T) / mat.shape[1]


def between_class_scatter_and_within_class_scatter(features, labels):
    num_classes = len(set(labels))
    mean_vector = vectorize_column(features.mean(1))
    Sb = 0
    Sw = 0
    for c in range(num_classes):
        nc = (labels[labels == c]).__len__()
        mean_c = vectorize_column(features[:, labels == c].mean(1))
        el = nc * (mean_c - mean_vector) * (mean_c - mean_vector).T
        Sb += el
        deviation_from_mean_c = features[:, labels == c] - mean_c
        Sw_c = np.dot(deviation_from_mean_c, deviation_from_mean_c.T)
        Sw += Sw_c
    Sb = Sb / labels.shape[0]
    Sw = Sw / labels.shape[0]
    return (Sb, Sw, num_classes)


def logpdf_multivariate_gaussian_nd(x, mu, C):
    M = x.shape[0]
    _, logs = np.linalg.slogdet(C)
    siginv = np.linalg.inv(C)
    rt = []
    for i in range(x.shape[1]):
        sample = vectorize_column(x[:, i])
        a = -M/2*np.log(2*np.pi) - 1/2*logs - 1/2 * \
            np.dot(np.dot((sample-mu).T, siginv), (sample-mu))
        rt.append(a)
    return np.array(rt).ravel()


def log_likelihood(x, mu, C):
    pdf = logpdf_multivariate_gaussian_nd(x, mu, C)
    return pdf.sum()


def create_confusion_matrix(predictions, labels, num_classes):
    CM = np.array([0] * num_classes**2).reshape(num_classes, num_classes)
    for i in range(predictions.size):
        CM[predictions[i], round(labels[i])] += 1
    return CM


def calculate_minimum_detection_cost(pi, cfn, cfp, scores, labels):
    thresholds = np.append(np.min(scores) - 1, scores)
    thresholds = np.append(thresholds, np.max(scores) + 1)
    thresholds.sort()
    DCFdummy = min(pi*cfn, (1-pi)*cfp)
    minDCF = 100000
    for t in thresholds:
        predictions = [1 if el > t else 0 for el in scores]
        CM = create_confusion_matrix(np.array(predictions), labels, 2)
        fnr = CM[0, 1] / (CM[1, 1] + CM[0, 1])
        fpr = CM[1, 0] / (CM[0, 0] + CM[1, 0])
        DCFu = pi*cfn*fnr + (1-pi)*cfp*fpr
        minDCF = (DCFu / DCFdummy) if (DCFu / DCFdummy) < minDCF else minDCF

    return minDCF


def perform_cross_validation(K, data_train, labels_train, num_classes, model, pca, pi, cfn=1, cfp=1, all_scores_condition=False):
    N = data_train.shape[1]
    seglen = N//K
    scores = []
    for i in range(K):
        mask = [True] * N
        mask[i*seglen: (i+1)*seglen] = [False] * seglen
        notmask = [not x for x in mask]
        data_train_resampled = data_train[:, mask]
        labels_train_resampled = labels_train[mask]
        data_validation = data_train[:, notmask]
        llr = model(data_train_resampled, labels_train_resampled,
                    data_validation, num_classes)
        scores += llr.tolist()

    print(pca, "minDCF:", calculate_minimum_detection_cost(
        pi, cfn, cfp, scores, labels_train))
    if all_scores_condition:
        return scores
    return calculate_minimum_detection_cost(pi, cfn, cfp, scores, labels_train)


def multivariate_gaussian_classifier(data_train, labels_train, data_validation, num_classes, priors=0):
    mus = []
    sigmas = []
    for i in range(num_classes):
        mus.append(mean_of_data_set(data_train[:, labels_train == i]))
        sigmas.append(covariance_matrix(data_train[:, labels_train == i]))

    S = []
    for j in range(data_validation.shape[1]):
        ccp = []
        for i in range(num_classes):
            ccp.append(np.exp(log_likelihood(
                vectorize_column(data_validation[:, j]), mus[i], sigmas[i])))
        S.append(vectorize_column(np.array(ccp)))
    S = np.hstack(S)

    if (priors == 0):
        priors = vectorize_column(np.array([1/num_classes] * num_classes))
    SJoint = priors * S
    SMarginal = vectorize_row(SJoint.sum(0))
    SPost = SJoint / SMarginal
    return np.log(SPost[1, :] / SPost[0, :])


def gaussian_naive_bayes(data_train, labels_train, data_validation, num_classes, priors=0):
    mus = []
    sigmas = []
    for i in range(num_classes):
        mus.append(mean_of_data_set(data_train[:, labels_train == i]))
        sigma = covariance_matrix(data_train[:, labels_train == i])
        sigmas.append(sigma * np.identity(sigma.shape[0]))

    S = []
    for j in range(data_validation.shape[1]):
        ccp = []
        for i in range(num_classes):
            ccp.append(np.exp(log_likelihood(
                vectorize_column(data_validation[:, j]), mus[i], sigmas[i])))
        S.append(vectorize_column(np.array(ccp)))
    S = np.hstack(S)

    if (priors == 0):
        priors = vectorize_column(np.array([1/num_classes] * num_classes))
    SJoint = priors * S
    SMarginal = vectorize_row(SJoint.sum(0))
    SPost = SJoint / SMarginal
    return np.log(SPost[1, :] / SPost[0, :])


def tied_gaussian_classifier(data_train, labels_train, data_validation, num_classes, priors=0):
    _, SW, _ = between_class_scatter_and_within_class_scatter(
        data_train, labels_train)
    mus = []
    for i in range(num_classes):
        mus.append(mean_of_data_set(data_train[:, labels_train == i]))

    S = []
    for j in range(data_validation.shape[1]):
        ccp = []
        for i in range(num_classes):
            ccp.append(np.exp(log_likelihood(
                vectorize_column(data_validation[:, j]), mus[i], SW)))
        S.append(vectorize_column(np.array(ccp)))
    S = np.hstack(S)

    if (priors == 0):
        priors = vectorize_column(np.array([1/num_classes] * num_classes))
    SJoint = priors * S
    SMarginal = vectorize_row(SJoint.sum(0))
    SPost = SJoint / SMarginal
    return np.log(SPost[1, :] / SPost[0, :])


def tied_naive_bayes_classifier(data_train, labels_train, data_validation, num_classes, priors=0):
    _, SW, _ = between_class_scatter_and_within_class_scatter(
        data_train, labels_train)
    SW = SW*np.identity(SW.shape[0])

    mus = []
    for i in range(num_classes):
        mus.append(mean_of_data_set(data_train[:, labels_train == i]))

    S = []
    for j in range(data_validation.shape[1]):
        ccp = []
        for i in range(num_classes):
            ccp.append(np.exp(log_likelihood(
                vectorize_column(data_validation[:, j]), mus[i], SW)))
        S.append(vectorize_column(np.array(ccp)))
    S = np.hstack(S)

    if (priors == 0):
        priors = vectorize_column(np.array([1/num_classes] * num_classes))
    SJoint = priors * S
    SMarginal = vectorize_row(SJoint.sum(0))
    SPost = SJoint / SMarginal
    return np.log(SPost[1, :] / SPost[0, :])


def run_gaussian_models(data, labels, pi, K, steps):
    print('\nMVG')
    perform_cross_validation(
        K, data, labels, 2, multivariate_gaussian_classifier, f"no PCA", pi)
    for i in range(steps):
        pt, _ = ml.PCA(data, 6-i)
        data_transformed = np.dot(pt, data)
        perform_cross_validation(
            K, data_transformed, labels, 2, multivariate_gaussian_classifier, f"PCA{6-i}", pi)

    print('\nNaive Bayes')
    perform_cross_validation(
        K, data, labels, 2, gaussian_naive_bayes, f"no PCA", pi)
    for i in range(steps):
        pt, _ = ml.PCA(data, 6-i)
        data_transformed = np.dot(pt, data)
        perform_cross_validation(
            K, data_transformed, labels, 2, gaussian_naive_bayes, f"PCA{6-i}", pi)

    print('\nTied gaussian')
    perform_cross_validation(
        K, data, labels, 2, tied_gaussian_classifier, f"no PCA", pi)
    for i in range(steps):
        pt, _ = ml.PCA(data, 6-i)
        data_transformed = np.dot(pt, data)
        perform_cross_validation(
            K, data_transformed, labels, 2, tied_gaussian_classifier, f"PCA{6-i}", pi)

    print('\nNaive tied gaussian')
    perform_cross_validation(
        K, data, labels, 2, tied_naive_bayes_classifier, f"no PCA", pi)
    for i in range(steps):
        pt, _ = ml.PCA(data, 6-i)
        data_transformed = np.dot(pt, data)
        perform_cross_validation(
            K, data_transformed, labels, 2, tied_naive_bayes_classifier, f"PCA{6-i}", pi)


if __name__ == "__main__":
    path = os.path.abspath("data/Train.txt")
    full_train_data, full_train_labels = load_data(path)

    K = 5
    steps = 2

    standard_deviation = np.std(full_train_data)
    z_data = ml.center_data(full_train_data) / standard_deviation

    pi = 0.1
    print('\n\ngaussian models with pi=0.1')
    run_gaussian_models(full_train_data, full_train_labels, pi, K, steps)
    print('\n\ngaussian models z-normalization pi=0.1')
    run_gaussian_models(z_data, full_train_labels, pi, K, steps)

    pi = 0.5
    print('\n\ngaussian models with pi=0.5')
    run_gaussian_models(full_train_data, full_train_labels, pi, K, steps)
    print('\n\ngaussian models z-normalization pi=0.5')
    run_gaussian_models(z_data, full_train_labels, pi, K, steps)
