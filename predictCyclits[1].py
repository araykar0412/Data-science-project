from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score as r2
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from matplotlib import pyplot as plt
import numpy as n


def normalize_train_y(y_train, Y):
    return (y_train - n.mean(Y, axis = 0)) / n.std(Y, axis=0), n.mean(Y, axis = 0), n.std(Y, axis=0)


def normalize_train(X_train):
    # normalizer = preprocessing.MinMaxScaler()
    X_norm = n.zeros(n.shape(X_train))

    mean = n.zeros(n.shape(X_train[0]))
    std = n.zeros(n.shape(X_train[0]))

    for i in range(3):
        X_norm[:, i] = (X_train[:, i] - n.mean(X_train[:, i])) / n.std(X_train[:, i])

        # X_norm[:, i] = normalizer.fit_transform(X_train[:, i])
        mean[i] = n.mean(X_train[:, i])
        std[i] = n.std(X_train[:, i])

    X = X_norm
    return X, mean, std
def normalize_test(X_test, trn_mean, trn_std):
    # fill in
    X_norm = n.zeros(n.shape(X_test))
    for i in range(3):
        X_norm[:, i] = (X_test[:, i] - trn_mean[i]) / trn_std[i]
    X = X_norm
    return X

def predictCyclists(df):
    X = df[['High Temp (°F)','Low Temp (°F)','Precipitation']].to_numpy()
    Y = df[['Total']].to_numpy()

    # fixing data to remove 1st day and move first day's number of cyclists to previous day
    # X = X[:-1]
    # Y = Y[1:]

    # making test and train data
    [X_train, X_test, y_train, y_test] = train_test_split(X, Y, train_size=.8, random_state=71)

    # normalize data
    [X_train, trn_mean, trn_std] = normalize_train(X_train)
    X_test = normalize_test(X_test, trn_mean, trn_std)

    [y_train, mean_y, std_y] = normalize_train_y(y_train, Y)

    # t
    l = n.logspace(-1, 2, num=201)

    MSE = []
    MODEL = []
    for lmbda in l:
        reg = Ridge(alpha = lmbda, fit_intercept=True)
        model = reg.fit(X_train, y_train)
        prediction = model.predict(X_test)
        prediction = prediction * std_y + mean_y
        # print(prediction)
        error = mse(prediction, y_test)
        MSE.append(error)
        MODEL.append(model)

    # print(MSE)
    bestMSE = min(MSE)
    bestLambda = l[MSE.index(bestMSE)]
    bestModel = MODEL[MSE.index(bestMSE)]
    bestResult = bestModel.predict(X_test) * std_y + mean_y
    r2Val = r2(y_test, bestResult)
    print(bestResult)
    print(bestLambda)
    print(len(bestResult))
    print(r2Val)
    print(bestMSE)
    print(bestModel.coef_)
    print(bestModel.intercept_)

    coefs = bestModel.coef_[0]
    # b0 = bestModel.intercept_[0]
    print(f'Model is as follows:\n\t Cyclists = {round(coefs[0], 2)} * High Temp (F) + {round(coefs[1], 2)} * Low Temp (F)  {round(coefs[2], 2)} * Precipitation')
    
    plt.plot(l, MSE)
    plt.title("MSE vs. Lambda")
    plt.xlabel(' Regularization parameter λ')
    plt.ylabel('Mean squared error')
    plt.show()
    

