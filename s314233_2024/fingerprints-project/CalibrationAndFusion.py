import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from bayesRisk import compute_minDCF_binary_fast, compute_actDCF_binary_fast
from LogisticRegression import trainWeightedLogRegBinary

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
    return effPriorLogOdds, actDCF, minDCF

def plot_bayes_error(log_odds_range, actual_DCF_uncal, min_DCF_uncal, actual_DCF_cal, min_DCF_cal, model_name):
    plt.figure(figsize=(10, 6))
    plt.plot(log_odds_range, actual_DCF_uncal, label=f'Actual DCF ({model_name}, Uncalibrated)', linestyle='--')
    plt.plot(log_odds_range, min_DCF_uncal, label=f'Min DCF ({model_name})', linestyle='-', color='red')
    plt.plot(log_odds_range, actual_DCF_cal, label=f'Actual DCF ({model_name}, Calibrated)', linestyle='-', color='blue')
    plt.xlabel('log-odds')
    plt.ylabel('DCF')
    plt.title(f'Bayes Error Plot for {model_name}')
    plt.legend()
    plt.grid(True)

    save_dir = os.path.join(os.getcwd(), "Image")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(os.path.join(save_dir, f"{model_name.replace(' ', '_')}_Calibrated_Bayes_Error_Plot.png"))
    
    plt.show()

# Generalized K-fold calibration function
def generalized_kfold_calibration(s_llr, L, training_priors, target_prior, model_name):   
    best_prior = None
    best_dcf = float('inf')
    best_w = None
    best_b = None
    
    
    for pT in training_priors:
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
        avg_dcf = compute_minDCF_binary_fast(calibrated_scores, labels_sys, target_prior, 1.0, 1.0)
        
        #print(f"Avg DCF for training prior {pT} is {avg_dcf:.4f}")
        
        if avg_dcf < best_dcf:
            best_dcf = avg_dcf
            best_prior = pT
            best_w = w
            best_b = b
            best_calibrated_scores = calibrated_scores
            best_labels_sys = labels_sys

    w_final, b_final = trainWeightedLogRegBinary(vrow(s_llr), L, 0, best_prior)

    np.save(f'Scores/final_w_{model_name}.npy', w_final)
    np.save(f'Scores/final_b_{model_name}.npy', b_final)
    
    print(f"Best training prior for {model_name}: {best_prior} with DCF: {best_dcf:.4f}")


    return best_prior, best_calibrated_scores, best_labels_sys

# Generalized K-fold fusion function for two models with Bayes plots
def generalized_kfold_fusion(s_llr_model1, s_llr_model2, L, training_priors, target_prior, model_name_1, model_name_2):
    best_dcf = float('inf')
    best_fused_scores = None
    best_prior = None
    best_w = None
    best_b = None


    for pT in training_priors:
        fused_scores = []
        fused_labels = []

        for foldIdx in range(KFOLD):
            SCAL_model1, SVAL_model1 = extract_train_val_folds_from_ary(s_llr_model1, foldIdx)
            SCAL_model2, SVAL_model2 = extract_train_val_folds_from_ary(s_llr_model2, foldIdx)
            LCAL, LVAL = extract_train_val_folds_from_ary(L, foldIdx)
            
            # Stack the two model scores
            SCAL = np.vstack([SCAL_model1, SCAL_model2])
            SVAL = np.vstack([SVAL_model1, SVAL_model2])

            w, b = trainWeightedLogRegBinary(SCAL, LCAL, 0, pT)
            
            calibrated_fused = (w.T @ SVAL + b - np.log(pT / (1 - pT))).ravel()
            fused_scores.append(calibrated_fused)
            fused_labels.append(LVAL)

        fused_scores = np.hstack(fused_scores)
        fused_labels = np.hstack(fused_labels)

        actual_DCF_fused = compute_actDCF_binary_fast(fused_scores, fused_labels, target_prior, 1.0, 1.0)
        
        if actual_DCF_fused < best_dcf:
            best_dcf = actual_DCF_fused
            best_fused_scores = fused_scores
            best_prior = pT
            best_w = w
            best_b = b

    # Generate Bayes plots for uncalibrated and calibrated fusion
    logOdds, actDCF_cal, minDCF_cal = bayesPlot(best_fused_scores, fused_labels)

    plot_bayes_error(logOdds, actDCF_cal, minDCF_cal, actDCF_cal, minDCF_cal, f"Fused {model_name_1} and {model_name_2}")

    # Retrain the final fusion model on the whole dataset
    SCAL_final = np.vstack([s_llr_model1, s_llr_model2])
    w_final, b_final = trainWeightedLogRegBinary(SCAL_final, L, 0, best_prior)
    
    np.save(f'Scores/calibrated_fused_scores_{model_name_1}_{model_name_2}.npy', best_fused_scores)
    np.save(f'Scores/final_w_fused_{model_name_1}_{model_name_2}.npy', w_final)
    np.save(f'Scores/final_b_fused_{model_name_1}_{model_name_2}.npy', b_final)
    
    print(f"Best training prior for fusion {model_name_1} with {model_name_2}: {best_prior} with DCF: {best_dcf:.4f}")


    return best_fused_scores, fused_labels, best_prior

# Generalized K-fold fusion function for three models with Bayes plots
def generalized_kfold_fusion_three(s_llr_quad, s_llr_gmm, s_llr_svm, L, training_priors, target_prior):
    best_dcf = float('inf')
    best_fused_scores = None
    best_prior = None
    best_w = None
    best_b = None

    for pT in training_priors:
        fused_scores = []
        fused_labels = []

        for foldIdx in range(KFOLD):
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

        actual_DCF_fused = compute_actDCF_binary_fast(fused_scores, fused_labels, target_prior, 1.0, 1.0)
        
        if actual_DCF_fused < best_dcf:
            best_dcf = actual_DCF_fused
            best_fused_scores = fused_scores
            best_prior = pT
            best_w = w
            best_b = b


    # Generate Bayes plots for uncalibrated and calibrated fusion
    logOdds_cal, actDCF_cal, minDCF_cal = bayesPlot(best_fused_scores, fused_labels)

    plot_bayes_error(logOdds_cal, actDCF_cal, minDCF_cal, actDCF_cal, minDCF_cal, "Fused of the three Models")

    # Retrain the final fusion model on the whole dataset
    SCAL_final = np.vstack([s_llr_quad, s_llr_gmm, s_llr_svm])
    w_final, b_final = trainWeightedLogRegBinary(SCAL_final, L, 0, best_prior)
    
    np.save('Scores/calibrated_fused_scores_quad_gmm_svm.npy', best_fused_scores)
    np.save(f'Scores/final_w_fused_quad_gmm_svm.npy', w_final)
    np.save(f'Scores/final_b_fused_quad_gmm_svm.npy', b_final)
    
    print(f"Best training prior for fusion of the three models: {best_prior} with DCF: {best_dcf:.4f}")


    return best_fused_scores, fused_labels, best_prior

# Function to generate Bayes plots
def generate_bayes_plots(s_llr, L, best_prior, model_name):
    logOdds_uncal, actDCF_uncal, minDCF_uncal = bayesPlot(s_llr, L)
    
    calibrated_scores = []
    labels_sys = []
    
    for foldIdx in range(KFOLD):
        DTR, DVAL = extract_train_val_folds_from_ary(s_llr, foldIdx)
        LTR, LVAL = extract_train_val_folds_from_ary(L, foldIdx)
        
        w, b = trainWeightedLogRegBinary(vrow(DTR), LTR, 0, best_prior)
        calibrated_llr = (w.T @ vrow(DVAL) + b - np.log(best_prior / (1 - best_prior))).ravel()
        
        calibrated_scores.append(calibrated_llr)
        labels_sys.append(LVAL)

    calibrated_scores = np.hstack(calibrated_scores)
    labels_sys = np.hstack(labels_sys)
    
    logOdds_cal, actDCF_cal, minDCF_cal = bayesPlot(calibrated_scores, labels_sys)
    
    plot_bayes_error(logOdds_uncal, actDCF_uncal, minDCF_uncal, actDCF_cal, minDCF_cal, model_name)

if __name__ == '__main__':
    path = 'Train.txt'
    D, L = load(path)
    (DTR, LTR), (DVAL, LVAL_original) = split_db_2to1(D, L)
    
    s_llr_svm = np.load('Scores/s_llr_svm.npy')
    s_llr_quad = np.load('Scores/s_llr_quad.npy')
    s_llr_gmm = np.load('Scores/s_llr_gmm.npy')
    
    # Shuffle 
    seed = 0  
    s_llr_svm, LVAL_training_svm = shuffle_data(s_llr_svm, LVAL_original, seed=seed)
    s_llr_quad, LVAL_training_quad = shuffle_data(s_llr_quad, LVAL_original, seed=seed)
    s_llr_gmm, LVAL_training_gmm = shuffle_data(s_llr_gmm, LVAL_original, seed=seed)
    
    target_prior = 0.1  
    training_priors = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]  
    
    # Best training prior for each model using the generalized K-fold approach
    best_prior_quad, calibrated_quad, labels_quad = generalized_kfold_calibration(s_llr_quad, LVAL_training_quad, training_priors, target_prior, "Quadratic Logistic Regression")
    best_prior_gmm, calibrated_gmm, labels_gmm = generalized_kfold_calibration(s_llr_gmm, LVAL_training_gmm, training_priors, target_prior, "GMM")
    best_prior_svm, calibrated_svm, labels_svm = generalized_kfold_calibration(s_llr_svm, LVAL_training_svm, training_priors, target_prior, "RBF SVM")

    # Generate Bayes plots
    generate_bayes_plots(s_llr_quad, LVAL_training_quad, best_prior_quad, "Quadratic Logistic Regression")
    generate_bayes_plots(s_llr_gmm, LVAL_training_gmm, best_prior_gmm, "GMM")
    generate_bayes_plots(s_llr_svm, LVAL_training_svm, best_prior_svm, "RBF SVM")

    # Fusion of the three models    
    fused_scores_three, fused_labels_three, _ = generalized_kfold_fusion_three(s_llr_quad, s_llr_gmm, s_llr_svm, LVAL_training_gmm, training_priors, target_prior)
    
    # Additional two-model fusions
    generalized_kfold_fusion(s_llr_quad, s_llr_svm, LVAL_training_quad, training_priors, target_prior, "Quadratic Logistic Regression", "RBF SVM")
    generalized_kfold_fusion(s_llr_gmm, s_llr_svm, LVAL_training_gmm, training_priors, target_prior, "GMM", "RBF SVM")
    generalized_kfold_fusion(s_llr_quad, s_llr_gmm, LVAL_training_gmm, training_priors, target_prior, "Quadratic Logistic Regression", "GMM")