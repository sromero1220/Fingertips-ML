import numpy as np
import matplotlib.pyplot as plt
from CalibrationAndFusion import compute_minDCF_binary_fast, compute_actDCF_binary_fast, plot_bayes_error

# Load the evaluation dataset
def load_evaluation_data(filepath):
    data = np.loadtxt(filepath, delimiter=',')
    X_eval = data[:, :-1].T  # Attributes
    L_eval = data[:, -1]     # Labels
    return X_eval, L_eval

# Evaluate the delivered system on the evaluation dataset
def evaluate_delivered_system(X_eval, L_eval, s_llr, target_prior, model_name):
    # Compute actual DCF
    actual_DCF = compute_actDCF_binary_fast(s_llr, L_eval, target_prior, 1.0, 1.0)
    # Compute minimum DCF
    min_DCF = compute_minDCF_binary_fast(s_llr, L_eval, target_prior, 1.0, 1.0)
    
    # Generate Bayes error plot
    log_odds_range = np.linspace(-4, 4, 21)
    plot_bayes_error(log_odds_range, actual_DCF, min_DCF, model_name)
    
    print(f"Evaluation of {model_name}:")
    print(f"  Actual DCF: {actual_DCF:.4f}")
    print(f"  Minimum DCF: {min_DCF:.4f}")

# Compare best performing systems and their fusion
def compare_best_systems(systems_scores, systems_labels, model_names, target_prior):
    for i, (s_llr, L) in enumerate(zip(systems_scores, systems_labels)):
        evaluate_delivered_system(None, L, s_llr, target_prior, model_names[i])
    
    # Fusion analysis (fusion of best three systems)
    fused_scores = np.mean(systems_scores, axis=0)
    evaluate_delivered_system(None, systems_labels[0], fused_scores, target_prior, "Fused System")

# Analyze the chosen calibration strategy for effectiveness
def analyze_calibration_strategy(X_eval, L_eval, models, model_names, target_prior):
    for model, model_name in zip(models, model_names):
        s_llr = model(X_eval)  # Assuming model is a function that returns scores
        evaluate_delivered_system(None, L_eval, s_llr, target_prior, model_name)

if __name__ == "__main__":
    eval_filepath = 'evalData.txt'
    target_prior = 0.1  
    
    # Load the evaluation data
    X_eval, L_eval = load_evaluation_data(eval_filepath)
    
    # Load the best models' scores
    s_llr_quad = np.load('Scores/s_llr_quad.npy')
    s_llr_gmm = np.load('Scores/s_llr_gmm.npy')
    s_llr_svm = np.load('Scores/s_llr_svm.npy')
    
    # Compare the best models and their fusion
    compare_best_systems([s_llr_quad, s_llr_gmm, s_llr_svm], [L_eval, L_eval, L_eval],
                         ["Quadratic Logistic Regression", "GMM", "SVM"], target_prior)
    
    # If additional models are available, analyze them
    # analyze_calibration_strategy(X_eval, L_eval, [model1, model2, model3], ["Model1", "Model2", "Model3"], target_prior)
