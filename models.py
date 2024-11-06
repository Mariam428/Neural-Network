# main.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tkinter import messagebox
import seaborn as sns
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
    df[['body_mass', 'beak_length', 'beak_depth', 'fin_length']]
)

# Adaline model class
class Adaline:
    def __init__(self, eta=0.01, epochs=1000, mse_threshold=0.01, add_bias=True):
        #eta--->learning rate
        self.eta = eta
        self.epochs = epochs
        self.mse_threshold = mse_threshold
        self.add_bias = add_bias
        self.weights = None

    def fit(self, X, y):
        if self.add_bias:
            X = np.c_[np.ones(X.shape[0]), X]  # Adding bias as the first column
        self.weights = np.random.uniform(-0.5, 0.5, X.shape[1])

        for epoch in range(self.epochs):
            #output=net_input = X×weights + bias
            # output = np.dot(X, self.weights) + self.bias

            output = self.net_input(X)
            errors = y - output
            self.weights += self.eta * X.T.dot(errors)
            if self.add_bias:
                self.bias += self.eta * errors.sum()
            mse = (errors ** 2).mean()
            print(f"Epoch: {epoch + 1}, MSE: {mse}")
            if mse <= self.mse_threshold:
                print("Training stopped due to MSE threshold.")
                break

    def net_input(self, X):
        return np.dot(X, self.weights)
    def predict(self, X):
        if self.add_bias:
            X = np.c_[np.ones(X.shape[0]), X]
        return np.dot(X, self.weights) + self.bias
        #return np.where(self.net_input(X) >= 0.0, 1, -1)

# Training functions
def train_adaline(eta, epochs, mse_threshold, add_bias, feature1, feature2, class1, class2):
    filtered_df = df[df['bird category'].isin([category_mapping[class1], category_mapping[class2]])]
    X = filtered_df[[feature1, feature2]].values
    y = np.where(filtered_df['bird category'] == category_mapping[class1], -1, 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

    adaline = Adaline(eta=eta, epochs=epochs, mse_threshold=mse_threshold, add_bias=add_bias)
    adaline.fit(X_train, y_train)

    test_adaline_model(adaline, X_test, y_test)


messagebox.showinfo("Training Completed", "Adaline training is complete.")
# Plot the decision boundary
   # plot_decision_boundary(adaline, X_train, y_train, title="Adaline Decision Boundary")

def test_adaline_model(adaline, X_test, y_test):
    # Predicting with the trained model
    y_pred = adaline.predict(X_test)

    # Confusion Matrix calculation
    TP = np.sum((y_test == 1) & (y_pred == 1))  # True Positive
    TN = np.sum((y_test == -1) & (y_pred == -1))  # True Negative
    FP = np.sum((y_test == -1) & (y_pred == 1))  # False Positive
    FN = np.sum((y_test == 1) & (y_pred == -1))  # False Negative

    cm = np.array([[TP, FP], [FN, TN]])

    # Accuracy calculation
    accuracy = np.sum(y_test == y_pred) / len(y_test) * 100

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.matshow(cm, cmap='Blues', alpha=0.5)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha='center', va='center', color='black')

    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Class 1', 'Class 2'])
    ax.set_yticklabels(['Class 1', 'Class 2'])
    plt.title('Confusion Matrix')
    plt.show()

def call_perceptron_train(eta, epochs, add_bias, feature1, feature2, class1, class2):
    class_mapping = {'A': 0, 'B': 1, 'C': 2}
    c = [class_mapping[class1], class_mapping[class2]]

    filtered_df = df[df['bird category'].isin(c)]
    X = filtered_df[[feature1, feature2]].values
    y = np.where(filtered_df['bird category'] == c[0], -1, 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

    #X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    #X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    weights = perceptron_algo_train(epochs, eta, int(add_bias), X_train, y_train)

def call_perceptron_train(eta, epochs, add_bias, feature1, feature2, class1, class2):
    print("in call perceptron train")
    class_mapping = {'A': 0, 'B': 1, 'C': 2}

    c=[0]*2
    c[0]= class_mapping[class1]
    c[1]=class_mapping[class2]

    #print(c)
    eta = float(eta)
    epochs=int(epochs)
    # Select features and filter the dataset based on selected classes
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
    calculate_confusion_matrix(test_df, f1, f2, weights, bias)
    # return  weights , test_df

def calculate_confusion_matrix(test_df, f1, f2, weights, bias):
    TP = TN = FP = FN = 0
    y_predict_test = [0] * len(test_df)
    y_predict_sign_test = [0] * len(test_df)

    # Iterate over the test data to make predictions
    for index, entry in test_df.iterrows():
        pos = test_df.index.get_loc(index)
        # Calculate the predicted value using the learned weights and bias
        y_predict_test[pos] = (entry[f1] * weights[0]) + (entry[f2] * weights[1]) + bias
        # Convert to the predicted class label
        y_predict_sign_test[pos] = np.sign(y_predict_test[pos])

        # Compare predictions with actual values for confusion matrix
        actual_label = int(entry['bird category'])
        predicted_label = int(y_predict_sign_test[pos])

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

    # Calculate accuracy from confusion matrix
    accuracy = (TP + TN) / (TP + TN + FP + FN) * 100

    # Plot confusion matrix as a heatmap
    cm = np.array([[TP, FP], [FN, TN]])  # Create the confusion matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted Negative', 'Predicted Positive'],
                yticklabels=['Actual Negative', 'Actual Positive'])
    plt.title(f'Confusion Matrix\nAccuracy: {accuracy:.2f}%')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()


# Visualization
def plot_decision_boundary(model, X, y, title="Decision Boundary"):
    # Scatter the points with different colors for each class
    plt.scatter(X[y == -1][:, 0], X[y == -1][:, 1], color='red', marker='o', label='Class -1')
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='blue', marker='x', label='Class 1')

    # Plot the decision boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx = np.linspace(x_min, x_max, 100)

    # Calculate the slope and intercept for the decision boundary line
    if model.add_bias:
        intercept = -model.weights[0] / model.weights[2]  # bias weight
        slope = -model.weights[1] / model.weights[2]  # feature weight
    else:
        intercept = 0
        slope = -model.weights[0] / model.weights[1]

    yy = slope * xx + intercept
    plt.plot(xx, yy, "k-", lw=2, label="Decision Boundary")

    # Customize plot
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title(title)
    plt.legend()
    plt.show()

