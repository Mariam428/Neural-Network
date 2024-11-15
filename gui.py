# gui.py
import tkinter as tk
from tkinter import ttk
from main import *


'''def on_train_button_click():
    eta = float(eta_entry.get())
    epochs = int(epochs_entry.get())
    mse_threshold = float(mse_threshold_entry.get())
    add_bias = bias_var.get()
    feature2 = feature2_var.get()
    class1 = classe1_var.get()
    class2 = classe2_var.get()

    if algorithm_var.get() == "Perceptron":
      global weightsP,biasP,test_df
      weightsP, biasP, test_df =train_perceptron(eta, epochs, add_bias,  feature2, class1, class2)


    else:
        global weightsA, biasA,X_test,y_test
        weightsA, biasA,X_test,y_test = train_adaline(eta, epochs, mse_threshold, add_bias, feature2, class1, class2)

def on_test_button_click():
    eta = float(eta_entry.get())
    epochs = int(epochs_entry.get())
    mse_threshold = float(mse_threshold_entry.get())
    add_bias = bias_var.get()
    class1 = classe1_var.get()
    class2 = classe2_var.get()

    if algorithm_var.get() == "Perceptron":
        test_perceptron(weightsP, biasP, test_df,  feature2, add_bias)
    else:
        test_adaline(weightsA, biasA, X_test, y_test, add_bias)'''

# Main window
root = tk.Tk()
root.title("Machine Learning GUI")
root.geometry("500x350")

ttk.Label(root, text="Number of hidden layers:").grid(row=0, column=0, padx=5, pady=5, sticky="e")
layers_entry = ttk.Entry(root)
layers_entry.insert(0, "1")
layers_entry.grid(row=0, column=1, padx=10, pady=5)


ttk.Label(root, text="Number of neurons in each layer:").grid(row=1, column=0, padx=5, pady=5, sticky="e")
layers_entry = ttk.Entry(root)
layers_entry.insert(0, "1")
layers_entry.grid(row=1, column=1, padx=10, pady=5)


# Learning rate
ttk.Label(root, text="Learning Rate (eta):").grid(row=4, column=0, padx=5, pady=5, sticky="e")
eta_entry = ttk.Entry(root)
eta_entry.insert(0, "0.001")
eta_entry.grid(row=4, column=1, padx=10, pady=5)

# Number of epochs
ttk.Label(root, text="Number of Epochs (m):").grid(row=5, column=0, padx=5, pady=5, sticky="e")
epochs_entry = ttk.Entry(root)
epochs_entry.insert(0, "10")
epochs_entry.grid(row=5, column=1, padx=10, pady=5)

# Bias checkbox
bias_var = tk.BooleanVar()
bias_checkbox = ttk.Checkbutton(root, text="Add Bias", variable=bias_var)
bias_checkbox.grid(row=7, column=1, padx=5, pady=5, sticky="w")

# Activation function selection
activation_var = tk.StringVar()
activation_combobox = ttk.Combobox(root, textvariable=activation_var)
activation_combobox['values'] = ["Sigmoid", "Hyperbolic Tangent sigmoid"]
activation_combobox.grid(row=8, column=1, padx=10, pady=5)
activation_combobox.current(0)

# Train button
'''train_button = ttk.Button(root, text="Train", command=on_train_button_click)
train_button.grid(row=9, column=1, pady=20)

test_button = ttk.Button(root, text="Test", command=on_test_button_click)
test_button.grid(row=10, column=1, padx=10)'''

root.mainloop()
