# main.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tkinter import messagebox
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

df = pd.read_csv('birds.csv')

# Data preprocessing
mode_gender = df['gender'][df['gender'] != 'NA'].mode()[0]
df['gender'] = df['gender'].replace('NA', mode_gender)

# Change categorical values to numerical
gender_mapping = {'male': 0, 'female': 1}
df['gender'] = df['gender'].map(gender_mapping)

category_mapping = {'A': 0, 'B': 1, 'C': 2}
df['bird category'] = df['bird category'].map(category_mapping)

# Scaling numerical data
scaler = StandardScaler()
df[['body_mass', 'beak_length', 'beak_depth', 'fin_length']] = scaler.fit_transform(
    df[['body_mass', 'beak_length', 'beak_depth', 'fin_length']])





'''def test_adaline(weights, bias, X_test, y_test, add_bias=True):
    y_pred = np.dot(X_test, weights) + bias
    y_pred_binary = np.where(y_pred >= 0.0, 1, -1)

    TP = 0
    TN = 0
    FP = 0
    FN = 0

    for i in range(len(y_test)):
        actual_label = y_test[i]
        predicted_label = y_pred_binary[i]
        actual_label = int(actual_label)

        if predicted_label == 1:
            if actual_label == 1:
                TP += 1
            else:
                FP += 1
        else:
            if actual_label == 1:
                FN += 1
            else:
                TN += 1

    # Create confusion matrix
    cm = np.array([[TP, FP], [FN, TN]])

    accuracy = (TP + TN) / (TP + TN + FP + FN) * 100
    messagebox.showinfo("Model Accuracy", f"Accuracy: {accuracy:.2f}%")

    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.matshow(cm, cmap='Reds', alpha=0.5)

    labels = [["TN", "FP"], ["FN", "TP"]]

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f'{labels[i][j]}\n{cm[i, j]}', ha='center', va='center', color='black')

    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Class 1', 'Class 2'])
    ax.set_yticklabels(['Class 1', 'Class 2'])
    plt.title('Confusion Matrix')
    plt.show()'''


