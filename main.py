import pandas as pd
import tkinter as tk
from sklearn.preprocessing import StandardScaler
from tkinter import ttk
import numpy as np
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

def call_perceptron_train():#this function only gets input from gui, maps them and call perceptron algo
    class_mapping = {'A': 0, 'B': 1, 'C': 2}
    c=[0,1,2]
    perceptron_algo_train(10, 0.1, 0, df, c, 0, 1)




def perceptron_algo_train(epochs,eta,bias,df,classes,f1_index,f2_index): #returns final weights and test df
    print("in perceptron_algo")
    #prepare the dataframe to work on
    new_df = df[df["bird category"] != classes[2]].copy() #drop unwanted class
    new_df = new_df.iloc[:, [int(f1_index), int(f2_index), -1]].copy() #select two features
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
            y_predict[pos] = (entry[int(f1_index)] * weights[0])+ (entry[int(f2_index)]* weights[1])+bias
            y_predict_sign[pos] = np.sign(y_predict[pos])
            if y_predict_sign[pos] != entry.iloc[-1]:
                loss= int(entry.iloc[-1]) - int(y_predict_sign[pos])
                #form new weights
                weights[0]=weights[0]+eta*loss*entry[int(f1_index)]
                weights[1]=weights[1]+eta *loss*entry[int(f2_index)]
                #print(f"loss  equals {loss}")

    print(f"final weights are: {weights[0]} and {weights[1]}")
    return  weights , test_df




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
#epochs_entry.insert(0,"10")
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
call_perceptron_train()
# Start the main event loop
root.mainloop()
