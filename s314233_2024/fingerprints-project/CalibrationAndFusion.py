import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from bayesRisk import compute_minDCF_binary_fast, compute_actDCF_binary_fast
from LogisticRegression import quadratic_expansion, trainLogRegBinary, trainWeightedLogRegBinary
from GMM import train_GMM_LBG_EM, logpdf_GMM
from SupportVectorMachines import train_dual_SVM_kernel, rbfKernel

KFOLD = 5
log_odds_range = np.linspace(-4, 4, 21)

def vrow(x):
    return x.reshape((1, x.size))

def vcol(x):
    return x.reshape((x.size, 1))

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

def extract_train_val_folds_from_ary(X, idx):
    return np.hstack([X[jdx::KFOLD] for jdx in range(KFOLD) if jdx != idx]), X[idx::KFOLD]

def shuffle_data(scores, labels, seed=None):
    if seed is not None:
        np.random.seed(seed)
    indices = np.random.permutation(scores.shape[0])
    return scores[indices], labels[indices]

def bayesPlot(S, L, left=-4, right=4, npts=21):
    effPriorLogOdds = np.linspace(left, right, npts)
    effPriors = 1.0 / (1.0 + np.exp(-effPriorLogOdds))
    actDCF = []
    minDCF = []
    for effPrior in effPriors:
        actDCF.append(compute_actDCF_binary_fast(S, L, effPrior, 1.0, 1.0))
        minDCF.append(compute_minDCF_binary_fast(S, L, effPrior, 1.0, 1.0))
    avg_separation = np.mean(np.array(actDCF) - np.array(minDCF))
    return effPriorLogOdds, actDCF, minDCF, avg_separation


def plot_bayes_error(log_odds_range, actual_DCF, min_DCF, model_name):
    plt.figure(figsize=(10, 6))
    plt.plot(log_odds_range, actual_DCF, label=f'Actual DCF ({model_name})', color='blue')
    plt.plot(log_odds_range, min_DCF, label=f'Min DCF ({model_name})', color='red')
    plt.xlabel('log-odds')
    plt.ylabel('DCF')
    plt.title(f'Bayes Error Plot for Calibrated {model_name}')
    plt.legend()
    plt.grid(True)
    save_dir = os.path.join(os.getcwd(), "Image")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(os.path.join(save_dir, f"{model_name.replace(' ', '_')}_Calibrated_Bayes_Error_Plot.png"))
    plt.show()

def evaluate_priors(s_llr, L, priors, target_prior, model_name):
    best_prior = None
    best_dcf = float('inf')

    
    for pT in priors:
        calibrated_scores = []
        labels_sys = [] 
        
        for foldIdx in range(KFOLD):
            DTR, DVAL = extract_train_val_folds_from_ary(s_llr, foldIdx)
            LTR, LVAL = extract_train_val_folds_from_ary(L, foldIdx)
            
            w, b = trainWeightedLogRegBinary(vrow(DTR), LTR, 0, pT)
            calibrated_llr = (w.T @ vrow(DVAL) + b - np.log(pT / (1-pT))).ravel()
            calibrated_scores.append(calibrated_llr)
            labels_sys.append(LVAL)
            
            
        calibrated_scores = np.hstack(calibrated_scores)
        labels_sys = np.hstack(labels_sys)
        avg_dcf=compute_minDCF_binary_fast(calibrated_scores, labels_sys, target_prior, 1.0, 1.0)
        
        print(f"Avg DCF for training prior {pT} is {avg_dcf:.4f}")
        
        if avg_dcf < best_dcf:
            best_dcf = avg_dcf
            best_prior = pT
            
    print(f"Best training prior for {model_name}: {best_prior} with DCF: {best_dcf:.4f}")
    return best_prior




def generate_bayes_plots(s_llr, L, best_prior, model_name):
    all_logOdds = []
    all_actDCF = []
    all_minDCF = []
    avg_separations = []

    for foldIdx in range(KFOLD):
        DTR, DVAL = extract_train_val_folds_from_ary(s_llr, foldIdx)
        LTR, LVAL = extract_train_val_folds_from_ary(L, foldIdx)
        
        w, b = trainWeightedLogRegBinary(vrow(DTR), LTR, 0, best_prior)
        calibrated_llr = (w.T @ vrow(DVAL) + b - np.log(best_prior / (1 - best_prior))).ravel()
        
        logOdds, actDCF, minDCF, avg_separation = bayesPlot(calibrated_llr, LVAL)
        all_logOdds.append(logOdds)
        all_actDCF.append(actDCF)
        all_minDCF.append(minDCF)
        avg_separations.append(avg_separation)

    plot_bayes_error(np.mean(all_logOdds, axis=0), np.mean(all_actDCF, axis=0), np.mean(all_minDCF, axis=0), model_name)
    avg_sep = np.mean(avg_separations)
    print(f"Average separation between actual and min DCF ({model_name}): {avg_sep:.8e}")

def perform_fusion(s_llr_quad, s_llr_gmm, s_llr_svm, L, training_priors, target_prior):
    best_dcf = float('inf')
    best_fused_scores = None
    best_prior = None

    for pT in training_priors:
        fused_scores = []
        fused_labels = []

        for foldIdx in range(KFOLD):
            # Prepare training data for fusion
            SCAL_quad, SVAL_quad = extract_train_val_folds_from_ary(s_llr_quad, foldIdx)
            SCAL_gmm, SVAL_gmm = extract_train_val_folds_from_ary(s_llr_gmm, foldIdx)
            SCAL_svm, SVAL_svm = extract_train_val_folds_from_ary(s_llr_svm, foldIdx)
            LCAL, LVAL = extract_train_val_folds_from_ary(L, foldIdx)
            
            SCAL = np.vstack([SCAL_quad, SCAL_gmm, SCAL_svm])
            SVAL = np.vstack([SVAL_quad, SVAL_gmm, SVAL_svm])

            w, b = trainWeightedLogRegBinary(SCAL, LCAL, 0, pT)
            calibrated_fused = (w.T @ SVAL + b - np.log(pT / (1 - pT))).ravel()
            fused_scores.append(calibrated_fused)
            fused_labels.append(LVAL)

        fused_scores = np.hstack(fused_scores)
        fused_labels = np.hstack(fused_labels)

        # Evaluate the fusion model using the target application prior
        actual_DCF_fused = compute_actDCF_binary_fast(fused_scores, fused_labels, target_prior, 1.0, 1.0)
        
        if actual_DCF_fused < best_dcf:
            best_dcf = actual_DCF_fused
            best_fused_scores = fused_scores
            best_prior = pT

    # Compute the minimum DCF and average separation for the best model
    min_DCF_fused = compute_minDCF_binary_fast(best_fused_scores, fused_labels, target_prior, 1.0, 1.0)
    logOdds, actDCF, minDCF, avg_separation = bayesPlot(best_fused_scores, fused_labels)
    
    print(f"Best training prior for fused model: {best_prior}")
    print(f"Actual DCF for best fused model: {best_dcf:.4f}")
    print(f"Min DCF for best fused model: {min_DCF_fused:.4f}")
    print(f"Average separation between actual and min DCF (Fused Model): {avg_separation:.8e}")

    # Plot fusion results for the best prior
    plot_bayes_error(log_odds_range, actDCF, minDCF, "Fused Model")

if __name__ == '__main__':
    path = 'Train.txt'
    D, L = load(path)
    (DTR, LTR), (DVAL, LVAL_original) = split_db_2to1(D, L)
    
    s_llr_svm = np.load('Scores/s_llr_svm.npy')
    s_llr_quad = np.load('Scores/s_llr_quad.npy')
    s_llr_gmm = np.load('Scores/s_llr_gmm.npy')
    
    # Shuffle scores -> similar distributions across folds
    seed = 0  
    s_llr_svm, LVAL_training = shuffle_data(s_llr_svm, LVAL_original, seed=seed)
    s_llr_quad, LVAL_training = shuffle_data(s_llr_quad, LVAL_original, seed=seed)
    s_llr_gmm, LVAL_training = shuffle_data(s_llr_gmm, LVAL_original, seed=seed)
    
    target_prior = 0.1  # Fixed target application prior for evaluation
    training_priors = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]  # List of training priors to test
    
    # Best training prior for each model
    best_prior_quad = evaluate_priors(s_llr_quad, LVAL_training, training_priors, target_prior, "Quadratic Logistic Regression")
    best_prior_gmm = evaluate_priors(s_llr_gmm, LVAL_training, training_priors, target_prior, "GMM")
    best_prior_svm = evaluate_priors(s_llr_svm, LVAL_training, training_priors, target_prior, "SVM")
    
    # Bayes plots 
    generate_bayes_plots(s_llr_quad, LVAL_training, best_prior_quad, "Quadratic Logistic Regression")
    generate_bayes_plots(s_llr_gmm, LVAL_training, best_prior_gmm, "GMM")
    generate_bayes_plots(s_llr_svm, LVAL_training, best_prior_svm, "SVM")
    
    # Fusion
    perform_fusion(s_llr_quad, s_llr_gmm, s_llr_svm, LVAL_training, training_priors, target_prior)
