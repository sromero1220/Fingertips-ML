import numpy as np
import pandas as pd
from GMM import load as load_gmm, logpdf_GMM, train_GMM_LBG_EM, split_db_2to1, evaluate_gmm_performance
from LogisticRegression import quadratic_expansion, trainLogRegBinary
from SupportVectorMachines import train_dual_SVM_kernel, rbfKernel 
from bayesRisk import compute_minDCF_binary_fast, compute_actDCF_binary_fast
import matplotlib.pyplot as plt
import os

def vrow(x):
    return x.reshape((1, x.size))

def load(filepath):
    df = pd.read_csv(filepath, header=None)
    attributes = np.array(df.iloc[:, :-1]).T
    labels = np.array(df.iloc[:, -1])
    return attributes, labels

def compute_model_scores(D_train, L_train, D, L, model_type, saved_model=False, pi_emp=0.1):
    """ Compute raw scores using the pre-trained models or saved scores. """
    
    if saved_model == False:
        (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D_train, L_train)
        if model_type == 'logistic_regression':
            
            D_quad_train = quadratic_expansion(DTR)  
            lambda_best_quad = 0.00031622776601683794
            w, b = trainLogRegBinary(D_quad_train, LTR, lambda_best_quad) 
            
            D_quad = quadratic_expansion(D)  
            scores = (np.dot(w.T, D_quad) + b).ravel()  
            s_llr = scores - np.log(pi_emp / (1 - pi_emp))  # LLR scores
            
            # Save scores for future use
            np.save(f"Scores/Evaluation_scores_{model_type}.npy", s_llr)
        
        elif model_type == 'gmm':
           
            num_components_best_gmm = 4
            gmm0 = train_GMM_LBG_EM(DTR[:, LTR == 0], num_components_best_gmm, covType='diagonal', verbose=False, psiEig=0.01)  # Train for class 0
            gmm1 = train_GMM_LBG_EM(DTR[:, LTR == 1], num_components_best_gmm, covType='diagonal', verbose=False, psiEig=0.01)  # Train for class 1
            
            scores = logpdf_GMM(D, gmm1) - logpdf_GMM(D, gmm0)  
            s_llr = scores - np.log(pi_emp / (1 - pi_emp))  # LLR scores

            # Save scores for future use
            np.save(f"Scores/Evaluation_scores_{model_type}.npy", s_llr)
        
        elif model_type == 'svm':
           
            gamma_best_svm = np.exp(-2)
            C_best_svm = 10 ** (3 / 2)
            rbf_kernel = rbfKernel(gamma_best_svm)  
            
            fScore_rbf = train_dual_SVM_kernel(DTR, LTR, C_best_svm, rbf_kernel, eps=1.0)  
            scores = fScore_rbf(D)  
            s_llr = scores - np.log(pi_emp / (1 - pi_emp))  # LLR scores

            # Save scores for future use
            np.save(f"Scores/Evaluation_scores_{model_type}.npy", s_llr)
        
        else:
            raise ValueError(f"Model type {model_type} is not supported.")
    else:
        # Load the previously saved scores
        s_llr = np.load(f"Scores/Evaluation_scores_{model_type}.npy")
    
    return s_llr  # Return the log-likelihood ratio scores for evaluation


def calibrate_scores(scores, w, b, prior=0.1):
    """ Calibrate scores using the saved weights and biases """
    calibrated_scores = (w.T @ vrow(scores) + b - np.log(prior / (1 - prior))).ravel()
    return calibrated_scores

def load_calibration_parameters(model_name):
    """ Load the precomputed calibration parameters w and b for each model """
    w = np.load(f"Scores/final_w_{model_name}.npy")
    b = np.load(f"Scores/final_b_{model_name}.npy")
    return w, b

def compute_evaluation_metrics(cal_scores, no_cal_scores, L, model_name):
    """ Compute minDCF and actDCF metrics """
    min_dcf = compute_minDCF_binary_fast(no_cal_scores, L, prior=0.1, Cfn=1.0, Cfp=1.0)
    act_dcf = compute_actDCF_binary_fast(no_cal_scores, L, prior=0.1, Cfn=1.0, Cfp=1.0)
    cal_act_dcf = compute_actDCF_binary_fast(cal_scores, L, prior=0.1, Cfn=1.0, Cfp=1.0)
    print(f"{model_name} - minDCF: {min_dcf:.4f}, actDCF: {act_dcf:.4f}, calibrated actDCF: {cal_act_dcf:.4f}")
    return min_dcf, act_dcf

def compute_evaluation_metrics_fusion(scores, L, model_name):
    """ Compute minDCF and actDCF metrics """
    min_dcf = compute_minDCF_binary_fast(scores, L, prior=0.1, Cfn=1.0, Cfp=1.0)
    act_dcf = compute_actDCF_binary_fast(scores, L, prior=0.1, Cfn=1.0, Cfp=1.0)
    print(f"{model_name} - minDCF: {min_dcf:.4f}, actDCF: {act_dcf:.4f}")
    return min_dcf, act_dcf

def bayes_plot(scores, labels, model_name, left=-3, right=3, npts=21):
    """ Generate and save a Bayes error plot """
    effPriorLogOdds = np.linspace(left, right, npts)
    effPriors = 1.0 / (1.0 + np.exp(-effPriorLogOdds))
    actDCF = []
    minDCF = []
    for effPrior in effPriors:
        actDCF.append(compute_actDCF_binary_fast(scores, labels, effPrior, 1.0, 1.0))
        minDCF.append(compute_minDCF_binary_fast(scores, labels, effPrior, 1.0, 1.0))

    plt.figure(figsize=(10, 6))
    plt.plot(effPriorLogOdds, actDCF, label=f'Actual DCF ({model_name})', color='blue')
    plt.plot(effPriorLogOdds, minDCF, label=f'Min DCF ({model_name})', color='red')
    plt.xlabel('log-odds')
    plt.ylabel('DCF')
    plt.title(f'Bayes Error Plot for {model_name}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"Image/{model_name.replace(' ', '_')}_Bayes_Error_Plot.png")
    plt.show()

def fuse_models(scores_list, w_fusion, b_fusion, prior=0.1):
    """ Fuse the calibrated scores using the fusion parameters """
    stacked_scores = np.vstack(scores_list)
    fused_scores = (w_fusion.T @ stacked_scores + b_fusion - np.log(prior / (1 - prior))).ravel()
    return fused_scores


def evaluate_chosen_gmm(DTR, LTR, DVAL, LVAL):
    num_components_best_gmm = 4
    print("\nEvaluating the chosen GMM model (diagonal covariance, 4 components):")
    
    gmm0 = train_GMM_LBG_EM(DTR[:, LTR == 0], num_components_best_gmm, covType='diagonal', verbose=False, psiEig=0.01)
    gmm1 = train_GMM_LBG_EM(DTR[:, LTR == 1], num_components_best_gmm, covType='diagonal', verbose=False, psiEig=0.01)
    
    SLLR = logpdf_GMM(DVAL, gmm1) - logpdf_GMM(DVAL, gmm0)
    
    minDCF = compute_minDCF_binary_fast(SLLR, LVAL, 0.1, 1.0, 1.0)
    print(f"Chosen GMM model (diagonal, 4 components) - minDCF: {minDCF:.4f}")
    
    return minDCF

def evaluate_other_gmm_configurations(DTR, LTR, DVAL, LVAL):
    num_components_list = [1, 2, 4, 8, 16, 32]
    cov_types = ['full', 'diagonal']

    print("\nEvaluating other GMM configurations:")
    results = evaluate_gmm_performance(DTR, LTR, DVAL, LVAL, num_components_list, cov_types)

    for cov_type in cov_types:
        for num_components, minDCF, actDCF in results[cov_type]:
            print(f"GMM ({cov_type}, {num_components} components) - minDCF: {minDCF:.4f}, actDCF: {actDCF:.4f}")
    
    return results
if __name__ == '__main__':
    path = 'Train.txt'
    D_train, L_train = load(path)
    evalDataPath = 'evalData.txt'
    D_eval, L_eval = load(evalDataPath)
    
    Load_training=False

    # Raw scores using the models
    pi_emp = (L_train == 1).sum() / L_train.size
    scores_quad = compute_model_scores(D_train, L_train,D_eval, L_eval, 'logistic_regression', saved_model=Load_training, pi_emp=pi_emp)
    scores_gmm = compute_model_scores(D_train, L_train,D_eval, L_eval, 'gmm', saved_model=Load_training, pi_emp=pi_emp)
    scores_svm = compute_model_scores(D_train, L_train,D_eval, L_eval, 'svm', saved_model=Load_training, pi_emp=pi_emp)

    # Calibration parameters 
    print("\nCalibrating the scores:")
    w_quad, b_quad = load_calibration_parameters('Quadratic Logistic Regression')
    calibrated_scores_quad = calibrate_scores(scores_quad, w_quad, b_quad, prior=0.9)

    w_gmm, b_gmm = load_calibration_parameters('GMM')
    calibrated_scores_gmm = calibrate_scores(scores_gmm, w_gmm, b_gmm, prior=0.7)

    w_svm, b_svm = load_calibration_parameters('RBF SVM')
    calibrated_scores_svm = calibrate_scores(scores_svm, w_svm, b_svm, prior=0.9)

    # Evaluation metrics for calibrated models
    print("\nEvaluating calibrated models:")
    compute_evaluation_metrics(calibrated_scores_quad, scores_quad, L_eval, "Quadratic Logistic Regression")
    compute_evaluation_metrics(calibrated_scores_gmm, scores_gmm, L_eval, "Diagonal Covariance GMM")
    compute_evaluation_metrics(calibrated_scores_svm, scores_svm, L_eval, "RBF SVM")

    # Bayes error plots for each model
    print("\nGenerating Bayes error plots:")
    bayes_plot(calibrated_scores_quad, L_eval, "Evaluation Calibrated Quadratic Logistic Regression")
    bayes_plot(calibrated_scores_gmm, L_eval, "Evaluation Calibrated Diagonal Covariance GMM")
    bayes_plot(calibrated_scores_svm, L_eval, "Evaluation Calibrated RBF SVM")

    # Fusion parameters
    print("\nFusing calibrated models:")
    
    # Fusion 1: Quad + GMM + SVM
    w_fusion_quad_gmm_svm = np.load('Scores/final_w_fused_quad_gmm_svm.npy')
    b_fusion_quad_gmm_svm = np.load('Scores/final_b_fused_quad_gmm_svm.npy')
    fused_scores_quad_gmm_svm = fuse_models([calibrated_scores_quad, calibrated_scores_gmm, calibrated_scores_svm], w_fusion_quad_gmm_svm, b_fusion_quad_gmm_svm, prior=0.8)
    compute_evaluation_metrics_fusion(fused_scores_quad_gmm_svm, L_eval, "Evaluation Fused Models (Quad + GMM + SVM)")
    bayes_plot(fused_scores_quad_gmm_svm, L_eval, "Evaluation Fused Models (Quad + GMM + SVM)")

    # Fusion 2: Quad + GMM
    w_fusion_quad_gmm = np.load('Scores/final_w_fused_Quadratic Logistic Regression_GMM.npy')
    b_fusion_quad_gmm = np.load('Scores/final_b_fused_Quadratic Logistic Regression_GMM.npy')
    fused_scores_quad_gmm = fuse_models([calibrated_scores_quad, calibrated_scores_gmm], w_fusion_quad_gmm, b_fusion_quad_gmm, prior=0.1)
    compute_evaluation_metrics_fusion(fused_scores_quad_gmm, L_eval, "Evaluation Fused Models (Quad + GMM)")
    bayes_plot(fused_scores_quad_gmm, L_eval, "Evaluation Fused Models (Quad + GMM)")

    # Fusion 3: Quad + SVM
    w_fusion_quad_svm = np.load('Scores/final_w_fused_Quadratic Logistic Regression_RBF SVM.npy')
    b_fusion_quad_svm = np.load('Scores/final_b_fused_Quadratic Logistic Regression_RBF SVM.npy')
    fused_scores_quad_svm = fuse_models([calibrated_scores_quad, calibrated_scores_svm], w_fusion_quad_svm, b_fusion_quad_svm, prior=0.1)
    compute_evaluation_metrics_fusion(fused_scores_quad_svm, L_eval, "Evaluation Fused Models (Quad + SVM)")
    bayes_plot(fused_scores_quad_svm, L_eval, "Evaluation Fused Models (Quad + SVM)")

    # Fusion 4: GMM + SVM
    w_fusion_gmm_svm = np.load('Scores/final_w_fused_GMM_RBF SVM.npy')
    b_fusion_gmm_svm = np.load('Scores/final_b_fused_GMM_RBF SVM.npy')
    fused_scores_gmm_svm = fuse_models([calibrated_scores_gmm, calibrated_scores_svm], w_fusion_gmm_svm, b_fusion_gmm_svm, prior=0.9)
    compute_evaluation_metrics_fusion(fused_scores_gmm_svm, L_eval, "Evaluation Fused Models (GMM + SVM)")
    bayes_plot(fused_scores_gmm_svm, L_eval, "Evaluation Fused Models (GMM + SVM)")
    
    
    #Evaluation of other configurations of GMM:
    
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D_train, L_train)


    chosen_model_dcf = evaluate_chosen_gmm(DTR, LTR, D_eval, L_eval)
    gmm_results = evaluate_other_gmm_configurations(DTR, LTR, D_eval, L_eval)

    # Find the best GMM configuration
    best_dcf = float('inf')
    best_config = None
    for cov_type in gmm_results:
        for num_components, minDCF, _ in gmm_results[cov_type]:
            if minDCF < best_dcf:
                best_dcf = minDCF
                best_config = (cov_type, num_components)

    # Display the best GMM configuration and its DCF
    print(f"Best GMM configuration: {best_config[0]} covariance with {best_config[1]} components")
    print(f"Best minDCF: {best_dcf:.4f}")

    if chosen_model_dcf == best_dcf:
        print("The chosen GMM model was optimal for the evaluation data.")
    else:
        print(f"There were other GMM configurations with better performance (Best minDCF: {best_dcf:.4f}).")

    

