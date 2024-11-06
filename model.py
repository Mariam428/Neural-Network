# main.py
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from tkinter import messagebox
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

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
    df[['body_mass', 'beak_length', 'beak_depth', 'fin_length']])

def train_adaline(eta, epochs, mse_threshold, add_bias, feature1, feature2, class1, class2):
    filtered_df = df[df['bird category'].isin([category_mapping[class1], category_mapping[class2]])]

    df_class1 = filtered_df[filtered_df['bird category'] == category_mapping[class1]]
    df_class2 = filtered_df[filtered_df['bird category'] == category_mapping[class2]]

    X_class1 = df_class1[[feature1, feature2]].values
    y_class1 = np.ones(X_class1.shape[0])  # Assign class 1 label

    X_class2 = df_class2[[feature1, feature2]].values
    y_class2 = -np.ones(X_class2.shape[0])  # Assign class 2 label

    X_train_class1, X_test_class1, y_train_class1, y_test_class1 = train_test_split(X_class1, y_class1, test_size=0.4,
                                                                                    random_state=42)
    X_train_class2, X_test_class2, y_train_class2, y_test_class2 = train_test_split(X_class2, y_class2, test_size=0.4,
                                                                                    random_state=42)

    X_train = np.vstack((X_train_class1, X_train_class2))
    y_train = np.concatenate((y_train_class1, y_train_class2))

    X_test = np.vstack((X_test_class1, X_test_class2))
    y_test = np.concatenate((y_test_class1, y_test_class2))

    if add_bias:
        X_train = np.insert(X_train, 0, 1, axis=1)
        X_test = np.insert(X_test, 0, 1, axis=1)

    weights = np.random.uniform(-0.5, 0.5, X_train.shape[1])
    bias = 0
    for epoch in range(epochs):
        output = np.dot(X_train, weights) + bias
        errors = y_train - output
        weights += eta * X_train.T.dot(errors)
        bias += eta * errors.sum()
        mse = (errors ** 2).mean()
        if mse <= mse_threshold:
            print("Training stopped due to MSE threshold.")
            break
    return weights, bias ,X_test,y_test

def test_adaline(weights, bias, X_test, y_test, add_bias=True):
    y_pred = np.dot(X_test, weights) + bias
    y_pred_binary = np.where(y_pred >= 0.0, 1, -1)
    # Initialize confusion matrix components
    TP = 0  # True Positive
    TN = 0  # True Negative
    FP = 0  # False Positive
    FN = 0  # False Negative

    # Loop over each test sample and calculate confusion matrix values
    for i in range(len(y_test)):
        actual_label = y_test[i]
        predicted_label = y_pred_binary[i]
        actual_label=int(actual_label)
        print("predicted_label", predicted_label)
        print("actual_label", actual_label)

        if predicted_label == 1:  # Predicted positive class
            if actual_label == 1:  # True positive
                TP += 1
            else:  # False positive
                FP += 1
        else:  # Predicted negative class
            if actual_label == 1:  # False negative
                FN += 1
            else:  # True negative
                TN += 1

    # Create confusion matrix
    cm = np.array([[TP, FP], [FN, TN]])

    # Calculate accuracy from confusion matrix
    accuracy = (TP + TN) / (TP + TN + FP + FN) * 100
    messagebox.showinfo("Model Accuracy", f"Accuracy: {accuracy:.2f}%")

    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.matshow(cm, cmap='Blues', alpha=0.5)

    # Define the labels based on position
    labels = [["TN", "FP"], ["FN", "TP"]]

    # Display the values along with the labels in each cell
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
    plt.show()

def train_perceptron(eta, epochs,add_bias, f1, f2, class1, class2):
    bias=0
    classes = [0] * 2
    classes[0] = category_mapping[class1]
    classes[1] = category_mapping[class2]

    #prepare the dataframe to work on
    new_df = df[(df["bird category"] == classes[0]) | (df["bird category"] == classes[1])].copy()
    new_df = new_df[[str(f1), str(f2),"bird category"]] #select two features

    #train test split
    class1_df = new_df[new_df["bird category"] ==  int(classes[0])]
    class2_df = new_df[new_df["bird category"] == int(classes[1])]

    # Randomly sample 30 entries from each class for training
    train_class1 = class1_df.sample(n=30, random_state=42)
    train_class2 = class2_df.sample(n=30, random_state=42)

    # Combine training data
    train_df = pd.concat([train_class1, train_class2])

    test_df = new_df[~new_df.index.isin(train_df.index)]

    weights = np.random.uniform(-0.5, 0.5, 2)
    y_predict =  [0] * len(train_df)
    y_predict_sign=  [0] * len(train_df)
    for i in range(epochs):
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

    X = train_df[[f1, f2]].values
    y = np.where(train_df["bird category"] == classes[0], -1, 1)  # Convert labels to -1 and 1
    plot_decision_boundary(weights, bias, X, y, add_bias,title="Perceptron Decision Boundary")
    return weights,bias,test_df


def test_perceptron(weights, bias, test_df, f1,f2,add_bias=True):
    X_test = test_df[[str(f1), str(f2)]].values
    y_test = test_df["bird category"].values

    y_pred = np.dot(X_test, weights) + bias

    y_pred_binary = np.where(y_pred >= 0.0, 1, -1)

    TP = TN = FP = FN = 0

    for i in range(len(y_test)):
        predicted_label = y_pred_binary[i]
        actual_label = y_test[i]
        # print("predicted_label", predicted_label)
        # print("actual_label", actual_label)

        if predicted_label == 1:  # Predicted positive class
            if actual_label == 1:  # True positive
                TP += 1
            else:  # False positive
                FP += 1
        else:  # Predicted negative class
            if actual_label == 1:  # False negative
                FN += 1
            else:  # True negative
                TN += 1

    cm = np.array([[TP, FP], [FN, TN]])

    accuracy = (TP + TN) / (TP + TN + FP + FN) * 100
    messagebox.showinfo("Model Accuracy", f"Accuracy: {accuracy:.2f}%")

    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.matshow(cm, cmap='Blues', alpha=0.5)

    # Define the labels based on position
    labels = [["TN", "FP"], ["FN", "TP"]]

    # Display the values along with the labels in each cell
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
    plt.show()


def plot_decision_boundary(weights, bias, X, y, add_bias=False, title="Decision Boundary"):

    plt.scatter(X[y == -1][:, 0], X[y == -1][:, 1], color='red', marker='o', label='Class -1')
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='blue', marker='x', label='Class 1')

    # Plot the decision boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx = np.linspace(x_min, x_max, 100)
    print(weights)
    print(bias)
    if add_bias:
        intercept = -bias / weights[2]
        slope = -weights[1] / weights[2]
    else:
        intercept = 0
        slope = -weights[0] / weights[1]

    yy = slope * xx + intercept
    plt.plot(xx, yy, "k-", lw=2, label="Decision Boundary")

    # Customize plot
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title(title)
    plt.legend()
    plt.show()
