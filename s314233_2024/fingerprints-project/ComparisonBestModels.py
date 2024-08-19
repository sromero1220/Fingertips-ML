import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin_l_bfgs_b
from bayesRisk import compute_minDCF_binary_fast, compute_actDCF_binary_fast
from GMM import load, split_db_2to1, train_GMM_LBG_EM, logpdf_GMM
from SupportVectorMachines import train_dual_SVM_kernel, rbfKernel
from LogisticRegression import quadratic_expansion, trainLogRegBinary, compute_metrics
import os

# Set the log-odds range
log_odds_range = np.linspace(-4, 4, 21)

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
    plt.title(f'Bayes Error Plot for {model_name}')
    plt.legend()
    plt.grid(True)
    save_dir = os.path.join(os.getcwd(), "Image")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(os.path.join(save_dir, f"{model_name.replace(' ', '_')}_Bayes_Error_Plot.png"))
    
    plt.show()


if __name__ == '__main__':
    path = 'Train.txt'
    D, L = load(path)
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)

    # Logistic Regression (Quadratic)
    pi_emp = (LTR == 1).sum() / LTR.size
    DTR_quad = quadratic_expansion(DTR)
    DVAL_quad = quadratic_expansion(DVAL)
    lambda_best_quad = 0.00031622776601683794  
    w, b = trainLogRegBinary(DTR_quad, LTR, lambda_best_quad)
    scores = (np.dot(w.T, DVAL_quad) + b).ravel()
    s_llr_quad = scores - np.log(pi_emp / (1 - pi_emp))
    
    np.save('Scores/s_llr_quad.npy', scores)

    _, actual_DCF_quad, min_DCF_quad, avg_separation_quad = bayesPlot(scores, LVAL)
    plot_bayes_error(log_odds_range, actual_DCF_quad, min_DCF_quad, "Quadratic Logistic Regression")
    print(f"Average separation between actual and min DCF (Quadratic Logistic Regression): {avg_separation_quad:.8e}")

    # GMM (Diagonal Covariance)
    num_components_best_gmm = 4  
    gmm0 = train_GMM_LBG_EM(DTR[:, LTR == 0], num_components_best_gmm, covType='diagonal', verbose=False, psiEig=0.01)
    gmm1 = train_GMM_LBG_EM(DTR[:, LTR == 1], num_components_best_gmm, covType='diagonal', verbose=False, psiEig=0.01)

    SLLR_gmm = logpdf_GMM(DVAL, gmm1) - logpdf_GMM(DVAL, gmm0)
    
    np.save('Scores/s_llr_gmm.npy', SLLR_gmm)
    
    _, actual_DCF_gmm, min_DCF_gmm, avg_separation_gmm = bayesPlot(SLLR_gmm, LVAL)
    plot_bayes_error(log_odds_range, actual_DCF_gmm, min_DCF_gmm, "GMM (Diagonal Covariance)")
    print(f"Average separation between actual and min DCF (GMM): {avg_separation_gmm:.8e}")

    # SVM (RBF Kernel)
    gamma_best_svm = np.exp(-2)  
    C_best_svm = 10**(3/2) 
    rbf_kernel = rbfKernel(gamma_best_svm)
    fScore_rbf = train_dual_SVM_kernel(DTR, LTR, C_best_svm, rbf_kernel, eps=1.0)
    s_llr_svm = fScore_rbf(DVAL)
    
    np.save('Scores/s_llr_svm.npy', s_llr_svm)

    _, actual_DCF_svm, min_DCF_svm, avg_separation_svm = bayesPlot(s_llr_svm, LVAL)
    plot_bayes_error(log_odds_range, actual_DCF_svm, min_DCF_svm, "SVM (RBF Kernel)")
    print(f"Average separation between actual and min DCF (SVM): {avg_separation_svm:.8e}")

   