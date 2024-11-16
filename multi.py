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

def training_phase(hidden_layers, neurons, eta, epochs, add_bias, activation):
    # initializing weights
    weights = []

    first_array = np.random.randn(5, neurons[0])
    weights.append(first_array)
    for i in range(0, len(neurons)):
        if i == len(neurons)-1:
            array = np.random.randn(neurons[i], 3)
        else:
            array = np.random.randn(neurons[i], neurons[i + 1])

        weights.append(array)

    # splitting the data for train and test
    train_df = df.groupby("bird category").apply(lambda x: x.sample(n=30, random_state=42))
    train_df = train_df.reset_index(drop=True)
    test_df = df[~df.index.isin(train_df.index)]

    # start training
    for m in range(epochs):
        for index, row in train_df.iterrows():
            x = train_df.iloc[index, 0:5].values.tolist()
            z = train_df.iloc[index, 5]
            y_hidden = []
            for i in range(0, hidden_layers):
                arr = []
                for j in range(0, neurons[i]):
                    if i == 0:
                        n = np.dot(weights[i][:, j], x)
                        if activation == "Sigmoid":
                            n = 1 / (1 + np.exp(-n))
                        elif activation == "Hyperbolic Tangent sigmoid":
                            n = np.tanh(n)
                        arr.append(n)
                    else:
                        n = np.dot(weights[i][:, j], y_hidden[i - 1])
                        if activation == "Sigmoid":
                            n = 1 / (1 + np.exp(-n))
                        elif activation == "Hyperbolic Tangent sigmoid":
                            n = np.tanh(n)
                        arr.append(n)
                y_hidden.append(arr)
            output = []
            for j in range(0, 3):
                l=len(weights)-1
                n = np.dot(weights[l][:, j], y_hidden[l - 1])
                if activation == "Sigmoid":
                    n = 1 / (1 + np.exp(-n))
                elif activation == "Hyperbolic Tangent sigmoid":
                    n = np.tanh(n)
                output.append(n)


    return test_df