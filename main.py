# main.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tkinter import messagebox

# Load the dataset
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
    df[['body_mass', 'beak_length', 'beak_depth', 'fin_length']]
)

# Adaline model class
class Adaline:
    def __init__(self, eta=0.01, epochs=1000, mse_threshold=0.01, add_bias=True):
        self.eta = eta
        self.epochs = epochs
        self.mse_threshold = mse_threshold
        self.add_bias = add_bias
        self.weights = None

    def fit(self, X, y):
        if self.add_bias:
            X = np.c_[np.ones(X.shape[0]), X]  # Adding bias as the first column
        self.weights = np.zeros(X.shape[1])

        for epoch in range(self.epochs):
            output = self.net_input(X)
            errors = y - output
            self.weights += self.eta * X.T.dot(errors)
            mse = (errors ** 2).mean()
            print(f"Epoch: {epoch + 1}, MSE: {mse}")
            if mse < self.mse_threshold:
                print("Training stopped due to MSE threshold.")
                break

    def net_input(self, X):
        return np.dot(X, self.weights)

    def predict(self, X):
        if self.add_bias:
            X = np.c_[np.ones(X.shape[0]), X]
        return np.where(self.net_input(X) >= 0.0, 1, -1)

# Training functions
def train_adaline(eta, epochs, mse_threshold, add_bias, feature1, feature2, class1, class2):
    filtered_df = df[df['bird category'].isin([category_mapping[class1], category_mapping[class2]])]
    X = filtered_df[[feature1, feature2]].values
    y = np.where(filtered_df['bird category'] == category_mapping[class1], -1, 1)

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    adaline = Adaline(eta=eta, epochs=epochs, mse_threshold=mse_threshold, add_bias=add_bias)
    adaline.fit(X_train, y_train)

    val_mse = ((y_val - adaline.predict(X_val)) ** 2).mean()
    print(f"Validation MSE: {val_mse}")

    messagebox.showinfo("Training Completed", "Adaline training is complete.")

def call_perceptron_train(eta, epochs, add_bias, feature1, feature2, class1, class2):
    class_mapping = {'A': 0, 'B': 1, 'C': 2}
    c = [class_mapping[class1], class_mapping[class2]]

    filtered_df = df[df['bird category'].isin(c)]
    X = filtered_df[[feature1, feature2]].values
    y = np.where(filtered_df['bird category'] == c[0], -1, 1)

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    perceptron_algo_train(epochs, eta, int(add_bias), X_train, y_train)

def perceptron_algo_train(epochs, eta, bias, X_train, y_train):
    weights = np.random.uniform(-0.5, 0.5, 2)
    y_predict = np.zeros(len(y_train))

    for i in range(epochs):
        for index, entry in enumerate(X_train):
            y_predict[index] = (entry[0] * weights[0]) + (entry[1] * weights[1]) + bias
            y_predict_sign = np.sign(y_predict[index])
            if y_predict_sign != y_train[index]:
                loss = int(y_train[index]) - int(y_predict_sign)
                weights[0] += eta * loss * entry[0]
                weights[1] += eta * loss * entry[1]

    print(f"Final weights for Perceptron are: {weights[0]} and {weights[1]}")
