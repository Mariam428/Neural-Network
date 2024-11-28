import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tkinter import messagebox

df = pd.read_csv('birds.csv')

# Data preprocessing
mode_gender = df['gender'][df['gender'] != 'NA'].mode()[0]
df['gender'] = df['gender'].fillna(mode_gender)

# Change categorical values to numerical
gender_mapping = {'male': 0, 'female': 1}
df['gender'] = df['gender'].map(gender_mapping)

category_mapping = {'A': 0, 'B': 1, 'C': 2}
df['bird category'] = df['bird category'].map(category_mapping)

# Scaling numerical data
scaler = StandardScaler()
df[['body_mass', 'beak_length', 'beak_depth', 'fin_length']] = scaler.fit_transform(
    df[['body_mass', 'beak_length', 'beak_depth', 'fin_length']])

# Splitting the data for train and test
# First, split the data by bird category
train_dfs = []
test_dfs = []

# Iterate through each category and sample 30 for training, 20 for testing
for category in df["bird category"].unique():
    category_df = df[df["bird category"] == category]

    # Sample 30 rows for the training set
    train_sample = category_df.sample(n=30, random_state=42)

    # The remaining 20 rows will be the test set
    test_sample = category_df.drop(train_sample.index)

    # Append to the lists
    train_dfs.append(train_sample)
    test_dfs.append(test_sample)

# Concatenate the sampled dataframes for train and test
train_df = pd.concat(train_dfs).reset_index(drop=True)
test_df = pd.concat(test_dfs).reset_index(drop=True)
#(test_df)


# data = {
#     'Col1': [3],
#     'Col2': [1],
#     'Col3': [2],
#     'Col4': [1],
#     'Col5': [2]
# }
# train_df = pd.DataFrame(data)

# print("Weights between input and hidden layer:")
# print(weights[0])
# for i in range(1, len(weights) - 1):
#     print(f"Weights between hidden layer {i} and hidden layer {i + 1}:")
#     print(weights[i])
# print("Weights between hidden and output layer:")
# print(weights[-1])
def forward_step(x, weights, hidden_layers, neurons, add_bias, activation):
    y_hidden = []
    for i in range(hidden_layers):
        arr = []
        for j in range(neurons[i]):
            if i == 0:  # Input layer to first hidden layer
                if add_bias:
                    n = np.dot(np.append(x, 1), weights[i][:, j])
                else:
                    n = np.dot(x, weights[i][:, j])
                if activation == "Sigmoid":
                    n = 1 / (1 + np.exp(-n))
                elif activation == "Hyperbolic Tangent sigmoid":
                    n = np.tanh(n)
                arr.append(n)
            else:
                # Hidden Layers * Weights
                if add_bias:
                    # adding the bias (ONE) to the hidden Layer
                    n = np.dot(np.append(y_hidden[i - 1], 1), weights[i][:, j])
                else:
                    n = np.dot(y_hidden[i - 1], weights[i][:, j])
                if activation == "Sigmoid":
                    n = 1 / (1 + np.exp(-n))
                elif activation == "Hyperbolic Tangent sigmoid":
                    n = np.tanh(n)
                arr.append(n)
        y_hidden.append(arr)
    return y_hidden


def backward_step(output, z, y_hidden, weights, hidden_layers, neurons, activation):
    # Output layer error
    sk = []
    z = np.array(z)
    for j in range(3):
        error = z[j] - output[j]
        if activation == "Sigmoid":
            derivative = output[j] * (1 - output[j])
        elif activation == "Hyperbolic Tangent sigmoid":
            derivative = 1 - (output[j] ** 2)
        sk.append(error * derivative)
    # print("output error :" , sk)

    # hidden layer error
    sh = []
    for i in range(hidden_layers - 1, -1, -1):
        s = np.zeros(neurons[i])
        for j in range(neurons[i]):  # For each neuron in the current layer
            if i == hidden_layers - 1:  # If it's the last hidden layer
                for k, error in enumerate(sk):  # Use output layer errors
                    s[j] += error * weights[i + 1][j, k]
            else:  # For other hidden layers
                for k, error in enumerate(sh[hidden_layers - i - 2]):
                    s[j] += error * weights[i + 1][j, k]
            if activation == "Sigmoid":
                s[j] *= y_hidden[i][j] * (1 - y_hidden[i][j])
            elif activation == "Hyperbolic Tangent sigmoid":
                s[j] *= (1 - (y_hidden[i][j] ** 2))

        sh.append(s)
    sh = sh[::-1]
    # print("hidden error :", sh)
    # updating weights
    # output layer

    return sk, sh


#print("**********************************************************************************")
# # print(train_df)
# print("hidden")
# print(y_hidden)
# print("f")
# print(output)
# print(sk)
# print(sh)
# print("weights after updatingggg")
# print("Weights between input and hidden layer:")
# print(weights[0])
# for i in range(1, len(weights) - 1):
#     print(f"Weights between hidden layer {i} and hidden layer {i + 1}:")
#     print(weights[i])
# print("Weights between hidden and output layer:")
# print(weights[-1])

# training_phase(3,[2,2,2],0.01,1,True,'Hyperbolic Tangent sigmoid')


def update_weights(weights, sk, sh, x, y_hidden, neurons, eta, hidden_layers, add_bias):
    # Update output layer weights
    l = len(weights) - 1
    for j in range(3):
        for i in range(len(weights[l]) - 1):
            weights[l][i, j] += eta * sk[j] * y_hidden[-1][i]
        if add_bias:
            weights[l][-1, j] += eta * sk[j]

    # the hidden layers
    for i in range(hidden_layers - 1, -1, -1):
        for j in range(neurons[i]):
            for k in range(len(weights[i]) - 1):
                if i == 0:
                    weights[i][k, j] += eta * sh[i][j] * x[k]
                else:
                    weights[i][k, j] += eta * sh[i][j] * y_hidden[i - 1][k]
            if add_bias:
                weights[i][-1, j] += eta * sh[i][j]
    return weights

weights = []
def training_phase(hidden_layers, neurons, eta, epochs, add_bias, activation):
    num_of_input = 5
    global weights
    weights = []
    correct_predictions = 0  # Initialize a counter for correct predictions
    total_predictions = 0    # Initialize a counter for total predictions

    if add_bias:
        num_of_input += 1

    # Input Layer -----> First Hidden Layer
    first_array = np.random.uniform(-1,1,size=(num_of_input, neurons[0]))
    weights.append(first_array)

    # Hidden Layer except the LAST ONE
    for i in range(0, len(neurons) - 1):
        if add_bias:
            array = np.random.uniform(-1,1,size=(neurons[i] + 1, neurons[i + 1]))
        else:
            array = np.random.uniform(-1,1,size=(neurons[i], neurons[i + 1]))
        weights.append(array)

    # Last Hidden Layer ----> Output Layer
    if add_bias:
        output_array = np.random.randn(neurons[-1] + 1, 3)
    else:
        output_array = np.random.randn(neurons[-1], 3)
    weights.append(output_array)

    for m in range(epochs):
        for index, row in train_df.iterrows():
            # Input
            x = train_df.iloc[index, 0:5].values.tolist()
            # Target
            z = np.zeros(3)
            z[train_df.iloc[index, -1]] = 1

            # Forward step
            y_hidden = forward_step(x, weights, hidden_layers, neurons, add_bias, activation)

            # Calculate output
            output = []
            for j in range(3):
                l = len(weights) - 1
                if add_bias:
                    n = np.dot(np.append(y_hidden[l - 1], 1), weights[l][:, j])
                else:
                    n = np.dot(y_hidden[l - 1], weights[l][:, j])

                if activation == "Sigmoid":
                    n = 1 / (1 + np.exp(-n))
                elif activation == "Hyperbolic Tangent sigmoid":
                    n = np.tanh(n)
                output.append(n)

            predicted_class = np.argmax(output)
            target_class = np.argmax(z)

            if predicted_class == target_class:
                correct_predictions += 1
            total_predictions += 1

            # Backward step
            sk, sh = backward_step(output, z, y_hidden, weights, hidden_layers, neurons, activation)

            # Update weights
            weights = update_weights(weights, sk, sh, x, y_hidden, neurons, eta, hidden_layers, add_bias)

    training_accuracy = (correct_predictions / total_predictions) * 100
    messagebox.showinfo("Model Accuracy", f"Training Accuracy: {training_accuracy:.2f}%")
    return weights


# training_phase(1,[2],0.001,1,True,'Hyperbolic Tangent sigmoid')

def testing_phase(hidden_layers, neurons, eta, epochs, add_bias, activation):
    predictions=[]
    true_labels=test_df['bird category'].values.tolist()
    #print("weights: ", weights)
    global weights
    for index, row in test_df.iterrows():
        x = row[test_df.columns[:-1]].values.tolist()
        y_hidden=forward_step(x, weights, hidden_layers, neurons, add_bias, activation)

        output=[]
        for j in range(3):
            l=len(weights)-1
            if add_bias:
                n=np.dot(np.append(y_hidden[l-1], 1), weights[l][:, j])
            else:
                n=np.dot(y_hidden[l-1], weights[l][:, j])

            if activation == "Sigmoid":
                n=1 / (1 + np.exp(-n))
            elif activation == "Hyperbolic Tangent sigmoid":
                n = np.tanh(n)
            output.append(n)

        #print(output)
        #predict based on maximum output
        predicted_class=np.argmax(output)
        # print(x)
        # print(output)
        # print(predicted_class)
        predictions.append(predicted_class)
        #print(predictions)
    #calc confusion matrix
    num_classes = 3
    cm = np.zeros((num_classes, num_classes), dtype=int)

    for true_label, predicted_label in zip(true_labels, predictions):
        cm[true_label, predicted_label] += 1

    #Plot confusion matrix
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.matshow(cm, cmap='Blues', alpha=0.7)

    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(j, i, f'{cm[i, j]}', ha='center', va='center', color='black')

    ax.set_xlabel('Predicted Labels', fontsize=12)
    ax.set_ylabel('True Labels', fontsize=12)
    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    ax.set_xticklabels([f'Class {i}' for i in range(num_classes)])
    ax.set_yticklabels([f'Class {i}' for i in range(num_classes)])
    plt.title('Confusion Matrix')
    plt.show()

    #accuracy
    total_correct = np.trace(cm)
    total_samples = len(true_labels)
    accuracy = (total_correct / total_samples) * 100

    messagebox.showinfo("Model Accuracy", f"Accuracy: {accuracy:.2f}%")