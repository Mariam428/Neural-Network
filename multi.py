import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

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


def training_phase(hidden_layers, neurons, eta, epochs, add_bias, activation):

    num_of_input = 5
    weights = []

    if add_bias:
        num_of_input += 1

    #Input Layer ----->First Hidden Layer
    first_array = np.random.randn(num_of_input, neurons[0])
    weights.append(first_array)

    #Hidden Layer expect the LAST ONE
    for i in range(0, len(neurons) - 1):
        if add_bias:
            array = np.random.randn(neurons[i] + 1, neurons[i + 1])
        else:
            array = np.random.randn(neurons[i], neurons[i + 1])
        weights.append(array)

    #Last Hidden Layer ----> Output Layer
    if add_bias:
        output_array = np.random.randn(neurons[-1] + 1, 3)
    else:
        output_array = np.random.randn(neurons[-1], 3)
    weights.append(output_array)

    # splitting the data for train and test
    train_df = df.groupby("bird category").apply(lambda x: x.sample(n=30, random_state=42))
    train_df = train_df.reset_index(drop=True)
    test_df = df[~df.index.isin(train_df.index)]

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

    for m in range(epochs):
        for index, row in train_df.iterrows():
            # input
            x = train_df.iloc[index, 0:5].values.tolist()
            # target
            z = train_df.iloc[index, -1]
            # hidden layers
            y_hidden = []
            for i in range(hidden_layers):
                arr = []
                for j in range(neurons[i]):
                    #Input * Weights
                    if i == 0:
                        if add_bias:
                            #adding the bias(ONE) to the input (x)
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
                            #adding the bias (ONE) to the hidden Layer
                            n = np.dot(np.append(y_hidden[i - 1], 1), weights[i][:, j])
                        else:
                            n = np.dot(y_hidden[i - 1], weights[i][:, j])
                        if activation == "Sigmoid":
                            n = 1 / (1 + np.exp(-n))
                        elif activation == "Hyperbolic Tangent sigmoid":
                            n = np.tanh(n)
                        arr.append(n)
                y_hidden.append(arr)

            output = []
            for j in range(3):
                l = len(weights) - 1
                if add_bias:
                    # adding the bias (ONE) to the output Layer
                    n = np.dot(np.append(y_hidden[l - 1], 1), weights[l][:, j])
                else:
                    n = np.dot(y_hidden[l - 1], weights[l][:, j])

                if activation == "Sigmoid":
                    n = 1 / (1 + np.exp(-n))
                elif activation == "Hyperbolic Tangent sigmoid":
                    n = np.tanh(n)
                output.append(n)
                # Backward step
                z = np.zeros(3)
                z[train_df.iloc[index, -1]] = 1

                # output layer error
                sk = []
                for j in range(3):
                    s = 0
                    if activation == "Sigmoid":
                        s = (z[j] - output[j]) * output[j] * (1 - output[j])
                    elif activation == "Hyperbolic Tangent sigmoid":
                        s = (z[j] - output[j])*(1 - (output[j]*output[j]))
                    sk.append(s)
                # hidden layer error
                sh = [None] * hidden_layers
                for i in range(hidden_layers - 1, -1, -1):
                    s = np.zeros(neurons[i])
                    for j in range(neurons[i]):
                        if i == hidden_layers - 1:
                            for k, error in enumerate(sk):
                                s[j] += error * weights[i + 1][j, k]
                        else:
                            for k, error in enumerate(sh[i + 1]):
                                s[j] += error * weights[i + 1][j, k]
                        if activation == "Sigmoid":
                            s[j] *= y_hidden[i][j] * (1 - y_hidden[i][j])
                        elif activation == "Hyperbolic Tangent sigmoid":
                            s[j] *= (1 - (y_hidden[i][j] * y_hidden[i][j]))
                    sh[i] = s
            # updating weights
            # output layer
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

    # print("**********************************************************************************")
    # # print(train_df)
    # print("hidden")
    # print(y_hidden)
    # print("f")
    # print(output)
    return test_df
# training_phase(1,[2],0.001,1,True,'Hyperbolic Tangent sigmoid')
