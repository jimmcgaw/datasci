from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

import pandas as pd


def loadBostonDF():
    data, target = load_boston(True)
    bostonDF = pd.DataFrame(data, columns=load_boston().feature_names)
    bostonDF['label'] = target
    return bostonDF


# via dataframe, if you want column names
# bostonDF = loadBostonDF()
# xTrain, xTest, yTrain, yTest = train_test_split(
    # bostonDF.drop(['label'], axis=1), bostonDF['label'], test_size=0.3)

# try it raw
data, target = load_boston(True)
xTrain, xTest, yTrain, yTest = train_test_split(data, target, test_size=0.3)

model = LinearRegression()
model.fit(xTrain, yTrain)

r2 = model.score(xTest, yTest)
print("R^2: {}".format(str(r2)))