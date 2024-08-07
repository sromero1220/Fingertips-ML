import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def vcol(x):
    return x.reshape((x.size, 1))

def vrow(x):
    return x.reshape((1, x.size))

def compute_mu_C(D):
    mu = vcol(D.mean(1))
    C = ((D-mu) @ (D-mu).T) / float(D.shape[1])
    return mu, C

def logpdf_GAU_ND(X, mu, C):
    P = np.linalg.inv(C)
    return -0.5*X.shape[0]*np.log(np.pi*2) - 0.5*np.linalg.slogdet(C)[1] - 0.5 * ((X-mu) * (P @ (X-mu))).sum(0)


if __name__ == '__main__':
    # Load the dataset
    file_path = 'Train.txt'
    data = pd.read_csv(file_path, header=None).to_numpy()

    # Separate features and class labels
    features = data[:, :-1]
    labels = data[:, -1].astype(int)

    # Separate data by class
    class_0_data = features[labels == 0]
    class_1_data = features[labels == 1]

    # Dictionary to hold class data
    data_dict = {'Class 0': class_0_data, 'Class 1': class_1_data}

    for label, X in data_dict.items():
        # For each feature in the class data
        for i in range(X.shape[1]):
            feature_data = X[:, i]
            m_ML = np.mean(feature_data)
            C_ML = np.var(feature_data)

            # Plot the histogram
            plt.figure()
            plt.hist(feature_data, bins=50, density=True, alpha=0.6, color='g')
            
            # Plot the estimated Gaussian density using logpdf_GAU_ND
            XPlot = np.linspace(min(feature_data), max(feature_data), 1000)
            m = np.array([m_ML])
            C = np.array([[C_ML]])
            log_density = logpdf_GAU_ND(vrow(XPlot), vcol(m), C)
            density = np.exp(log_density)

            plt.plot(XPlot, density.ravel(), linewidth=2, color='r')

            plt.title(f'{label} - Feature {i+1}')
            plt.xlabel('Feature Value')
            plt.ylabel('Density')
            plt.show()

            # Prints
            print(f'{label}, Feature {i+1}:')
            print(f'Mean: {m_ML}')
            print(f'Variance: {C_ML}')
