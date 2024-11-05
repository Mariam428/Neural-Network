# gui.py
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from main import train_adaline, call_perceptron_train  # Import functions from main.py

def on_train_button_click():
    eta = float(eta_entry.get())
    epochs = int(epochs_entry.get())
    mse_threshold = float(mse_threshold_entry.get())
    add_bias = bias_var.get()
    feature1 = feature1_var.get()
    feature2 = feature2_var.get()
    class1 = classe1_var.get()
    class2 = classe2_var.get()

    if algorithm_var.get() == "Perceptron":
        call_perceptron_train(eta, epochs, add_bias, feature1, feature2, class1, class2)
    else:
        train_adaline(eta, epochs, mse_threshold, add_bias, feature1, feature2, class1, class2)

# Main window
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
epochs_entry.insert(0, "10")
epochs_entry.grid(row=5, column=1, padx=10, pady=5)

# MSE threshold
ttk.Label(root, text="MSE Threshold:").grid(row=6, column=0, padx=5, pady=5, sticky="e")
mse_threshold_entry = ttk.Entry(root)
mse_threshold_entry.insert(0, "0.01")
mse_threshold_entry.grid(row=6, column=1, padx=10, pady=5)

# Bias checkbox
bias_var = tk.BooleanVar()
bias_checkbox = ttk.Checkbutton(root, text="Add Bias", variable=bias_var)
bias_checkbox.grid(row=7, column=1, padx=5, pady=5, sticky="w")

# Algorithm selection
algorithm_var = tk.StringVar()
algorithm_combobox = ttk.Combobox(root, textvariable=algorithm_var)
algorithm_combobox['values'] = ["Adaline", "Perceptron"]
algorithm_combobox.grid(row=8, column=1, padx=10, pady=5)
algorithm_combobox.current(0)

# Train button
train_button = ttk.Button(root, text="Train", command=on_train_button_click)
train_button.grid(row=9, column=1, pady=20)

root.mainloop()
