import os
import sys
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from itertools import combinations
sys.path.append(os.path.abspath("MLandPattern"))
import MLandPattern as ML

# Constants 
class_label = ["Fake fingerprint", "Genuine fingerprint"]
alpha_val = 0.5

# Helper functions
def load(pathname, visualization=0):
    df = pd.read_csv(pathname, header=None)
    if visualization:
        print(df.head())
    attribute = np.array(df.iloc[:, 0 : len(df.columns) - 1]).T
    label = np.array(df.iloc[:, -1])
    return attribute, label

def histogram_1n(fakeFing, genuineFing, x_axis=""):
    plt.xlim((min(min(fakeFing), min(genuineFing)) - 1, max(max(fakeFing), max(genuineFing)) + 1))
    plt.hist(fakeFing, color="blue", alpha=alpha_val, label=class_label[0], density=True, bins=np.arange(fakeFing.min(), fakeFing.max()+2,1), edgecolor='grey')
    plt.hist(genuineFing, color="red", alpha=alpha_val, label=class_label[1], density=True, bins=np.arange(genuineFing.min(), genuineFing.max()+2,1), edgecolor='grey')
    plt.legend(class_label)
    plt.title(f"{1+int(x_axis)}")

# Main plotting functions
def plot_scatter(features, labels, label_names, transparency=1):
    for i, j in combinations(range(features.shape[0]), 2):
        plt.figure()
        plt.title(f"Feature {i} vs Feature {j}")
        for label_index in range(len(label_names)):
            x_values = features[i][labels == label_index]
            y_values = features[j][labels == label_index]
            plt.scatter(x_values, y_values, label=label_names[label_index], alpha=transparency)
        plt.show()

def graficar(attributes):
    attribute_names = [str(i) for i in range(attributes.shape[0])]
    values_histogram = {name: [attributes[i, labels == 0], attributes[i, labels == 1]] for i, name in enumerate(attribute_names)}

    for cont, (xk, xv) in enumerate(values_histogram.items(), start=1):
        histogram_1n(xv[0], xv[1], x_axis=xk)
        plt.savefig(os.path.join(os.getcwd(), f"Image/histogram-dim-{cont}.png"))
        plt.show()

def graf_LDA(attributes, lables):
    W, _ = ML.LDA1(attributes, lables, 1)
    LDA_attributes = np.dot(W.T, attributes)
    for i in range(1):
        plt.hist(LDA_attributes[i, labels == 0], color="blue", density=True, alpha=0.7, label='0', bins=45)
        plt.hist(LDA_attributes[i, labels == 1], color="red", density=True, alpha=0.7, label='1', bins=45)
        plt.legend()
        plt.xlabel(i) 
    plt.title("LDA Direction")
    plt.savefig(f"{os.getcwd()}/Image/LDA-direction.png")
    plt.show()

def graf_PCA(attributes, lables):
    fractions = []
    total_eig, _ = np.linalg.eigh(ML.covariance(attributes))
    total_eig = np.sum(total_eig)
    for i in range(attributes.shape[0]):
        if i == 0:
            continue
        _, reduces_attributes = ML.PCA(attributes, i)
        PCA_means, _ = np.linalg.eigh(ML.covariance(reduces_attributes))
        PCA_means = np.sum(PCA_means)
        fractions.append(PCA_means/total_eig)
    fractions.append(1)
    plt.plot(range(1, 7), fractions, marker='o')
    plt.plot(range(1, 7), [0.97]*6, '--')
    plt.grid(True)
    plt.ylabel('Fraction of the retained variance')
    plt.xlabel('Number of dimensions')
    plt.title("PCA variability analysis")
    plt.savefig(f"{os.getcwd()}/Image/PCA-Analysis.png")
    plt.show()

def graph_corr(attributes, labels):
    corr_attr = np.corrcoef(attributes)
    sns.heatmap(corr_attr, cmap='Greens')
    plt.title("Total attribute correlation")
    plt.savefig(f"{os.getcwd()}/Image/Dataset-correlation.png")
    plt.show()

    fakeFings = attributes[:, labels == 0]
    corr_attr = np.corrcoef(fakeFings)
    sns.heatmap(corr_attr, cmap='Blues')
    plt.title("Fake Fingerprint attribute correlation")
    plt.savefig(f"{os.getcwd()}/Image/fakeFing-correlation.png")
    plt.show()

    genuineFings = attributes[:, labels == 1]
    corr_attr = np.corrcoef(genuineFings)
    sns.heatmap(corr_attr, cmap="Reds")
    plt.title("Genuine Fingerprint attribute correlation")
    plt.savefig(f"{os.getcwd()}/Image/genuineFing-correlation.png")
    plt.show()

if __name__ == "__main__":
    path = os.path.abspath("data/Train.txt")
    [attributes, labels] = load(path)
    
    for i in range(attributes.shape[0]):
        print(f"max {i}: {max(attributes[i])}", end="\t")
        print(f"min {i}: {min(attributes[i])}")
    
    print(f"Attribute dimensions: {attributes.shape[0]}")
    print(f"Points on the dataset: {attributes.shape[1]}")
    print(f"Distribution of labels (0, 1): {labels[labels==0].shape}, {labels[labels==1].shape}")
    print(f"Possible classes: {class_label[0]}, {class_label[1]}")
    
    pt, _ = ML.PCA(attributes, 6)
    att6 = np.dot(pt, attributes)
    
    plot_scatter(att6, labels, ['1', '2', '3', '4', '5', '6'])
    graficar(attributes)
    graf_LDA(attributes, labels)
    graf_PCA(attributes, labels)
    graph_corr(attributes, labels)