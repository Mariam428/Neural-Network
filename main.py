import pandas as pd
import tkinter as tk
from sklearn.preprocessing import StandardScaler
from tkinter import ttk
import numpy as np
from tkinter import messagebox
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv('birds.csv')

mode_gender = df['gender'][df['gender'] != 'NA'].mode()[0]
df['gender'] = df['gender'].replace('NA', mode_gender)

# change categorical values to numerical
gender_mapping = {'male': 0, 'female': 1}
df['gender'] = df['gender'].map(gender_mapping)

category_mapping = {'A': 0, 'B': 1, 'C': 2}
df['bird category'] = df['bird category'].map(category_mapping)


# scaling to numerical data
scaler = StandardScaler()

df[['body_mass', 'beak_length', 'beak_depth', 'fin_length']] = scaler.fit_transform(
    df[['body_mass', 'beak_length', 'beak_depth', 'fin_length']]
)

#print(df)
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

def train_adaline():
    # Retrieve values from GUI entries
    eta = float(eta_entry.get())
    epochs = int(epochs_entry.get())
    mse_threshold = float(mse_threshold_entry.get())
    add_bias = bias_var.get()

    # Select features and filter the dataset based on selected classes
    feature1 = feature1_var.get()
    feature2 = feature2_var.get()
    class1 = classe1_var.get()
    class2 = classe2_var.get()

    # Filter the dataset to include only selected classes
    filtered_df = df[df['bird category'].isin([category_mapping[class1], category_mapping[class2]])]
    X = filtered_df[[feature1, feature2]].values
    y = np.where(filtered_df['bird category'] == category_mapping[class1], -1, 1)  # Binary labels for Adaline

    # Initialize and train Adaline
    adaline = Adaline(eta=eta, epochs=epochs, mse_threshold=mse_threshold, add_bias=add_bias)
    adaline.fit(X, y)

    messagebox.showinfo("Training Completed", "Adaline training is complete.")

def call_perceptron_train():
    print("In call perceptron train")
    class_mapping = {'A': 0, 'B': 1, 'C': 2}
    c1 = classe1_var.get()
    c2 = classe2_var.get()
    c = [class_mapping[c1], class_mapping[c2]]

    eta = float(eta_entry.get())
    epochs = int(epochs_entry.get())
    add_bias = bias_var.get()

    # Select features
    feature1 = feature1_var.get()
    feature2 = feature2_var.get()

    # Filter the dataset
    filtered_df = df[df['bird category'].isin(c)]
    X = filtered_df[[feature1, feature2]].values
    y = np.where(filtered_df['bird category'] == c[0], -1, 1)

    # Split the data into train, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Call the perceptron algorithm training function
    perceptron_algo_train(epochs, eta, int(add_bias), X_train, y_train, feature1, feature2)

def perceptron_algo_train(epochs, eta, bias, X_train, y_train, f1, f2):
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

    print(f"final weights are: {weights[0]} and {weights[1]}")

def on_train_button_click():
    print("Button clicked")
    if algorithm_var.get() == "Perceptron":
        call_perceptron_train()
    else:
        print("Adaline training started")
        train_adaline()



# main window
root = tk.Tk()
root.title("Machine Learning GUI")
root.geometry("500x500")

# Dropdown for features
ttk.Label(root, text="Select Features:").grid(row=0, column=0, padx=5, pady=5, sticky="e")
feature1_var = tk.StringVar()
feature1_combobox = ttk.Combobox(root, textvariable=feature1_var)
feature1_combobox['values'] = ['body_mass', 'beak_length', 'beak_depth', 'fin_length']
feature1_combobox.grid(row=0, column=1, padx=5, pady=5)

feature2_var = tk.StringVar()
feature2_combobox = ttk.Combobox(root, textvariable=feature2_var)
feature2_combobox['values'] = ['body_mass', 'beak_length', 'beak_depth', 'fin_length']
feature2_combobox.grid(row=0, column=2, padx=5, pady=5)

# Select classes
ttk.Label(root, text="Classes:").grid(row=1, column=0, padx=5, pady=5, sticky="e")
classe1_var = tk.StringVar(value="A")
classe1_combobox = ttk.Combobox(root, textvariable=classe1_var)
classe1_combobox['values'] = ['A', 'B', 'C']
classe1_combobox.grid(row=1, column=1, padx=5, pady=5)

classe2_var = tk.StringVar(value="B")
classe2_combobox = ttk.Combobox(root, textvariable=classe2_var)
classe2_combobox['values'] = ['A', 'B', 'C']
classe2_combobox.grid(row=1, column=2, padx=5, pady=5)

# Learning rate
ttk.Label(root, text="Learning Rate (eta):").grid(row=4, column=0, padx=5, pady=5, sticky="e")
eta_entry = ttk.Entry(root)
eta_entry.insert(0, "0.1")
eta_entry.grid(row=4, column=1, padx=10, pady=5)

# Number of epochs
ttk.Label(root, text="Number of Epochs (m):").grid(row=5, column=0, padx=5, pady=5, sticky="e")
epochs_entry = ttk.Entry(root)
epochs_entry.insert(0,"10")
epochs_entry.grid(row=5, column=1, padx=10, pady=5)

# MSE threshold
ttk.Label(root, text="MSE Threshold:").grid(row=6, column=0, padx=5, pady=5, sticky="e")
mse_threshold_entry = ttk.Entry(root)
mse_threshold_entry.insert(0,"0.01")
mse_threshold_entry.grid(row=6, column=1, padx=10, pady=5)

# Bias checkbox
bias_var = tk.BooleanVar()
bias_checkbox = ttk.Checkbutton(root, text="Add Bias", variable=bias_var)
bias_checkbox.grid(row=7, column=1, padx=5, pady=5)

# Select algorithm
algorithm_var = tk.StringVar(value="Adaline")
ttk.Label(root, text="Select Algorithm:").grid(row=8, column=0, padx=5, pady=5, sticky="e")
adaline_radio = ttk.Radiobutton(root, text='Adaline', variable=algorithm_var, value='Adaline')
adaline_radio.grid(row=8, column=1, padx=5, pady=5, sticky="w")
perceptron_radio = ttk.Radiobutton(root, text='Perceptron', variable=algorithm_var, value='Perceptron')
perceptron_radio.grid(row=8, column=2, padx=5, pady=5, sticky="w")

# Train button
train_button = ttk.Button(root, text="Train", command=on_train_button_click)
train_button.grid(row=9, column=1, padx=5, pady=20)

root.mainloop()
