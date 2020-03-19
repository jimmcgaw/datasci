from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

import numpy as np

# import pandas as pd

def get_last_column(npArr):
    return npArr[:, -1]


def get_x_column(npArr, column=1):
    return npArr[:, column-1]


def get_all_but_last_column(npArr):
    return npArr[:, :-1]


def get_game_data_and_labels():
    with open('./games-expand.csv', 'r') as f:
        # not using pandas due to its memory bloat
        # gamesDF = pd.read_csv(f)

        arr = np.genfromtxt('./games-expand.csv', delimiter=',')
        # delete column text labels row
        arr = np.delete(arr, 0, axis=0)
        y = get_last_column(arr)
        X = get_all_but_last_column(arr)
    return X, y

X, y = get_game_data_and_labels()

xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.3)

model = LogisticRegression()
model.fit(xTrain, yTrain)

accuracy = model.score(xTest, yTest)
print("Accuracy: {}".format(accuracy))

xProbabilities = get_x_column(model.predict_proba(xTest), column=2)
auc = roc_auc_score(yTest, xProbabilities)
print("AUC: {}".format(auc))
