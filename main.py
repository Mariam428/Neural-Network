import pandas as pd
import tkinter as tk
from sklearn.preprocessing import StandardScaler
from tkinter import ttk
from tkinter import messagebox

df = pd.read_csv('birds.csv')

# change categorical values to numerical
gender_mapping = {'male': 0, 'female': 1, 'NA': 2}
df['gender'] = df['gender'].map(gender_mapping)

category_mapping = {'A': 0, 'B': 1, 'C': 2}
df['bird category'] = df['bird category'].map(category_mapping)


# scaling to numerical data
scaler = StandardScaler()

df[['body_mass', 'beak_length', 'beak_depth', 'fin_length']] = scaler.fit_transform(
    df[['body_mass', 'beak_length', 'beak_depth', 'fin_length']]
)

#print(df)


# main window
root = tk.Tk()
root.title("Machine Learning GUI")
root.geometry("500x500")

# Select features
ttk.Label(root, text="Features:").grid(row=0, column=0, padx=5, pady=5, sticky="e")
feature1_var = tk.StringVar(value="Feature1")
feature1_entry = ttk.Entry(root, textvariable=feature1_var)
feature1_entry.grid(row=0, column=1, padx=2.5, pady=2.5)
feature2_var = tk.StringVar(value="Feature2")
feature2_entry = ttk.Entry(root, textvariable=feature2_var)
feature2_entry.grid(row=0, column=2, padx=2.5, pady=2.5)


# Select classes
ttk.Label(root, text="Classes:").grid(row=2, column=0, padx=5, pady=5, sticky="e")
classe1_var = tk.StringVar(value="A")
classe1_entry = ttk.Entry(root, textvariable=classe1_var)
classe1_entry.grid(row=2, column=1, padx=5, pady=5)
classe2_var = tk.StringVar(value="B")
classe2_entry = ttk.Entry(root, textvariable=classe2_var)
classe2_entry.grid(row=2, column=2, padx=5, pady=5)


# Learning rate
ttk.Label(root, text="Learning Rate (eta):").grid(row=4, column=0, padx=5, pady=5, sticky="e")
eta_entry = ttk.Entry(root)
eta_entry.grid(row=4, column=1, padx=10, pady=5)

# Number of epochs
ttk.Label(root, text="Number of Epochs (m):").grid(row=5, column=0, padx=5, pady=5, sticky="e")
epochs_entry = ttk.Entry(root)
epochs_entry.grid(row=5, column=1, padx=10, pady=5)

# MSE threshold
ttk.Label(root, text="MSE Threshold:").grid(row=6, column=0, padx=5, pady=5, sticky="e")
mse_threshold_entry = ttk.Entry(root)
mse_threshold_entry.grid(row=6, column=1, padx=10, pady=5)

# bias checkbox
bias_var = tk.BooleanVar()
bias_checkbox = ttk.Checkbutton(root, text="Add Bias", variable=bias_var)
bias_checkbox.grid(row=7, column=1, padx=5, pady=5, sticky="w")

# Algorithm radio buttons
ttk.Label(root, text="Choose Algorithm:").grid(row=8, column=0, padx=5, pady=5, sticky="e")
algorithm_var = tk.StringVar(value="Perceptron")
perceptron_rb = ttk.Radiobutton(root, text="Perceptron", variable=algorithm_var, value="Perceptron")
adaline_rb = ttk.Radiobutton(root, text="Adaline", variable=algorithm_var, value="Adaline")
perceptron_rb.grid(row=8, column=1, padx=5, pady=5, sticky="w")
adaline_rb.grid(row=8, column=2, padx=5, pady=5, sticky="w")

# Train Test buttons
button1 = ttk.Button(root, text="Train")
button1.grid(row=11, column=1, padx=10, pady=5, sticky="e")

button2 = ttk.Button(root, text="Test")
button2.grid(row=11, column=2, padx=10, pady=5, sticky="w")

# Start the main event loop
root.mainloop()
