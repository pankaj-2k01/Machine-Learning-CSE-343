import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
#from sklearn.utils.validation import _Estimator
from sklearn.model_selection import GridSearchCV

df = pd.DataFrame()
learning_rate = 0.1
iteration = 1000


def read_data():
    global df
    # reading data seperated by ;
    df = pd.read_csv('abalone.data', sep=',', header=None)


def preprocess():
    """
    Used for the Pre processing 
    Parameters
    ----------
    None

    Returns
    -------
    X_train, y_train, X_test, y_test 
    :numpy array of preprocessed data

    """
    read_data()
    np.random.seed(0)  # setting seed to 0
    df.sample(frac=1)  # to get the 100% data
    df[0] = df[0].replace('M', 1)  # replacing values with integer values
    df[0] = df[0].replace('F', 2)
    df[0] = df[0].replace('I', 3)
    X_pro = df.iloc[:, :-1].values  # getting all rows values and column-1
    Y_pro = df.iloc[:, -1].values  # getting all rows values and last column
    # print(X_pro)
    # print(Y_pro)
    return X_pro, Y_pro


def cal_RMSE(y, y_0):
    """
    Used for the Calculation of RMSE
    Parameters-None
    Returns-RMSE error

    """
    n = len(y)
    summation = np.sum((y-y_0)**2)
    rmse_error = (summation*(1/n))**0.5  # rmse =(1/n)sqrt(summation((y-yo)^2))
    return rmse_error


def cal_cost(X, y, th0):
    """
    Used for the Calculation of Loss
    Parameters-X,y,theta
    Returns-loss

    """
    n = len(y)
    summation = np.sum(np.power((X.dot(th0)-y), 2))
    cost = summation/(y*2)  # cost=1/2n*summation((hbeta(xi)-yi)^2)
    return cost


def gradient_descent(X, y):
    """
    Used for the Calculation of Gradient descent
    Parameters-X,y
    Returns-theta

    """
    global iteration
    global learning_rate
    # print(X)
    # print(X.shape)
    X=np.hstack((np.ones((X.shape[0],1)),X))
    th_0 = np.zeros((X.shape[1],))
    # print(th_0)
    # print(th_0.shape)
    n = len(y)
    for i in range(iteration+1):
        Z = np.dot(X, th_0)
        loss = (Z-y)
        weight = X.T.dot(loss) / n
        th_0 -= learning_rate*weight
        #cost = cal_cost(X, y, th_0)
        # print("Cost at ",i,"is",cost)
    return th_0


def prediction(X, th_0):
    """
    Used for the Calculation of prediction
    Parameters-X,theta
    Returns-X.theta (numpy array)

    """
    X=np.hstack((np.ones((X.shape[0],1)),X))
    return X.dot(th_0)


def Linear_Regression():
    """
    Driver Function for Linear Regression
    Parameters-None
    Returns-None

    """
    X,y=preprocess()
    n = X.shape[0]
    # splitting 80:20 training and testing data

    X_test = X[int(n*0.8):]
    Y_train = y[:int(n*0.8)]
    Y_test = y[int(n*0.8):]
    X_train = X[:int(n*0.8)]
    th_0 = gradient_descent(X_train, Y_train)
    print(th_0)

    Y0_train = prediction(X_train, th_0)
    Y0_test = prediction(X_test, th_0)
    train_rmse = cal_RMSE(Y_train, Y0_train)
    test_rmse = cal_RMSE(Y_test, Y0_test)
    print("Train RMSE = ", train_rmse, " Test RMSE = ", test_rmse)

Linear_Regression()
# ----------------------------------------Q1b----------------------
def Regularization():
    """
    Used for the Calculation of Alpha and Coefficient for both Ridge and Lasso regression
    Parameters-None
    Returns-RMSE error

    """
    X, y = preprocess()
    n = X.shape[0]
    X_train = X[:int(n*0.8)]
    Y_train = y[:int(n*0.8)]
    X_test = X[int(n*0.8):]
    Y_test = y[int(n*0.8):]
    alpha = [1e-05, 0.0001, 0.001, 0.0015, 0.02, 0.2, 1, 2, 8, 10]
    Ridge_rmse = []
    Lasso_rmse = []
    Ridge_coef=[]
    Lasso_coef=[]
    for i in alpha:
        rr = Ridge(alpha=i)
        lr = Lasso(alpha=i)
        rr.fit(X_train, Y_train)
        pred_test_rr = rr.predict(X_test)
        print(np.sqrt(mean_squared_error(Y_test, pred_test_rr)))
        Ridge_rmse.append(np.sqrt(mean_squared_error(Y_test, pred_test_rr)))
        Ridge_coef.append(np.hstack((rr.intercept_,rr.coef_)))
        #print(r2_score(Y_train, pred_train_rr))
        lr.fit(X_train, Y_train)
        pred_test_lr = lr.predict(X_test)
        print(np.sqrt(mean_squared_error(Y_test, pred_test_lr)))
        Lasso_rmse.append(np.sqrt(mean_squared_error(Y_test, pred_test_lr)))
        Lasso_coef.append(np.hstack((lr.intercept_,lr.coef_)))
        #print(r2_score(Y_test, pred_test_rr))
    print()
    min_ridge_rmse=min(Ridge_rmse)
    min_lasso_rmse=min(Lasso_rmse)

    min_ridge_coef=min(Ridge_rmse)
    min_lasso_coef=min(Lasso_rmse)


    print("Best Ridge alpha would be : ",alpha[Ridge_rmse.index(min_ridge_rmse)])
    print("Best Ridge Coef would be : ",Ridge_coef[Ridge_rmse.index(min_ridge_coef)])

    print("Best Lasso alpha would be : ",alpha[Lasso_rmse.index(min_lasso_rmse)])
    print("Best Lasso Coef would be : ",Lasso_coef[Lasso_rmse.index(min_lasso_coef)])

    """
    Just for plotting

    """

    plt.plot(alpha, Ridge_rmse, marker='*', label='Ridge Regression')
    plt.plot(alpha, Lasso_rmse, marker='X', label='Lasso Regression')
    plt.legend()
    plt.title('RMSE VS ALPHA')
    plt.xscale("log")
    plt.xlabel("alphas")
    plt.ylabel("RMSE")
    plt.show()


Regularization()

#-------------------Q1-2b------------
def grid_search():
    """
    Used for the Calculation of alpha for both Ridge and Lasso using GridSearchCV
    Parameters-None
    Returns-RMSE error

    """
    X, y = preprocess()
    n = X.shape[0]
    X_train = X[:int(n*0.8)]
    Y_train = y[:int(n*0.8)]
    X_test = X[int(n*0.8):]
    Y_test = y[int(n*0.8):]
    alphaa = [1e-05, 0.0001, 0.001, 0.0015, 0.02, 0.2, 1, 2, 8, 10]
    grid_ridge=GridSearchCV(estimator=Ridge(),param_grid=dict(alpha=alphaa))
    grid_lasso = GridSearchCV(
            estimator=Lasso(), param_grid=dict(alpha=alphaa))

    #fitting the data
    grid_ridge.fit(X,y)
    grid_lasso.fit(X,y)

    print("Best Ridge alpha would be : ",grid_ridge.best_estimator_.alpha)
    print("Best Rdige Coef would be : ",np.hstack((grid_ridge.best_estimator_.intercept_,grid_ridge.best_estimator_.coef_)))
    #print("efgewgwegfewf",np.insert((grid_ridge.best_estimator_.intercept_,grid_ridge.best_estimator_.coef_)))
    print("Best Lasso alpha would be : ",grid_lasso.best_estimator_.alpha)
    print("Best Lasso Coef would be : ",np.hstack((grid_lasso.best_estimator_.intercept_,grid_lasso.best_estimator_.coef_)))

grid_search()