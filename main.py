import pandas as pd
import tkinter as tk
from sklearn.preprocessing import StandardScaler
from tkinter import ttk
import numpy as np
from tkinter import messagebox

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

print(df)
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

def call_perceptron_train():#this function only gets input from gui, maps them and call perceptron algo
    print("in call perceptron train")
    class_mapping = {'A': 0, 'B': 1, 'C': 2}
    c1=classe1_entry.get()
    c2=classe2_entry.get()
    c=[0]*2
    c[0]= class_mapping[c1]
    c[1]=class_mapping[c2]

    #print(c)
    eta = eta_entry.get()
    eta = float(eta)
    epochs = epochs_entry.get()
    epochs=int(epochs)
    add_bias = bias_var.get()
    # Select features and filter the dataset based on selected classes
    feature1 = feature1_var.get()
    feature2 = feature2_var.get()
    perceptron_algo_train(epochs, eta, int(add_bias), df, c, feature1, feature2)
    return




def perceptron_algo_train(epochs,eta,bias,df,classes,f1,f2): #returns final weights and test df
    print("in perceptron_algo")
    #prepare the dataframe to work on
    new_df = df[(df["bird category"] == classes[0]) | (df["bird category"] == classes[1])].copy()
    print(new_df.columns)
    new_df = new_df[[str(f1), str(f2),"bird category"]] #select two features
    print(new_df)
    #train test split
    class1_df = new_df[new_df["bird category"] ==  int(classes[0])]
    class2_df = new_df[new_df["bird category"] == int(classes[1])]
    # Randomly sample 30 entries from each class for training
    train_class1 = class1_df.sample(n=30, random_state=42)
    train_class2 = class2_df.sample(n=30, random_state=42)
    # Combine training data
    train_df = pd.concat([train_class1, train_class2])
    train_df.dropna(inplace=True)
    # Get the remaining entries for testing #TESTING FOR SALAH
    test_df = new_df[~new_df.index.isin(train_df.index)]
    weights = np.random.uniform(-0.5, 0.5, 2)
    y_predict =  [0] * len(train_df)
    y_predict_sign=  [0] * len(train_df)
    for i in range(epochs):
        #print(f"in epoch number {i}")
        for index, entry in train_df.iterrows():
            pos = train_df.index.get_loc(index)
            # Calculate y_predict for the current entry
            y_predict[pos] = (entry[f1] * weights[0])+ (entry[f2]* weights[1])+bias
            y_predict_sign[pos] = np.sign(y_predict[pos])
            if y_predict_sign[pos] != entry.iloc[-1]:
                loss= int(entry.iloc[-1]) - int(y_predict_sign[pos])
                #form new weights
                weights[0]=weights[0]+eta*loss*entry[f1]
                weights[1]=weights[1]+eta *loss*entry[f2]
                #print(f"loss  equals {loss}")

    print(f"final weights are: {weights[0]} and {weights[1]}")
    return  weights , test_df
def on_train_button_click():
    print("button clicked")
    if algorithm_var.get() == "Perceptron":
        call_perceptron_train()
    else:
        print("Adaline training started")
        train_adaline()



# main window
root = tk.Tk()
root.title("Machine Learning GUI")
root.geometry("500x500")

# Select features
ttk.Label(root, text="Features:").grid(row=0, column=0, padx=5, pady=5, sticky="e")
feature1_var = tk.StringVar(value="1")
feature1_entry = ttk.Entry(root, textvariable=feature1_var)
feature1_entry.grid(row=0, column=1, padx=2.5, pady=2.5)
feature2_var = tk.StringVar(value="2")
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
mse_threshold_entry.insert(0,"0.0")
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
button1 = ttk.Button(root, text="Train",command=on_train_button_click)
button1.grid(row=11, column=1, padx=10, pady=5, sticky="e")

button2 = ttk.Button(root, text="Test")
button2.grid(row=11, column=2, padx=10, pady=5, sticky="w")

# Start the main event loop
root.mainloop()
