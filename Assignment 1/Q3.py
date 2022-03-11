import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
#from sklearn.utils.validation import _Estimator
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import Binarizer
from numpy.core.fromnumeric import var
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import precision_recall_curve, recall_score, accuracy_score,confusion_matrix,roc_curve

df = pd.DataFrame()
def read_data():
    """
    Used for the reading the data
    Parameters-None
    Returns-None

    """
    global df
    # reading data seperated by ;
    df = pd.read_csv('fashion-mnist_test.csv', sep=',')
    # print(df.head)


def pre_process():
    """
    Used for the Pre-Processing 
    Parameters-None
    Returns- Training and Testing Data

    """
    np.random.seed(0)
    data_train = df.to_numpy()
    # print(df_train.head)
    data_test = df.to_numpy()
    f=np.logical_or(data_train[:, 0] == 1, data_train[:, 0] == 2)
    data_train = data_train[f]
    f1=np.logical_or(data_test[:, 0] == 1, data_test[:, 0] == 2)
    data_test = data_test[f1]
    X_train, y_train = data_train[:, 1:], data_train[:, 0]
    X_test, y_test = data_test[:, 1:], data_test[:, 0]
    binarizer = Binarizer(threshold=127)
    X_train = binarizer.fit_transform(X_train)
    # print(X_train)
    # y_train = binarizer.transform(y_train)
    X_test = binarizer.fit_transform(X_test)
    # y_test = binarizer.transform(y_test)
    return X_train, y_train, X_test, y_test
def GaussNB():
    """
    Used for the Calculation of Accuracy, Confusion Matrix 
    Parameters-None
    Returns- None

    """
    X_train,y_train,X_test,y_test=pre_process()
    g = GaussianNB()
    #fitting the data by
    g.fit(X_train, y_train)
    y_hat = g.predict(X_test)
    acc = accuracy_score(y_test, y_hat)
    #print(accur)
    #--------Accuracy, Confusion Matrix------
    precision, recall, thresholds = precision_recall_curve(y_test, y_hat,pos_label=2)
    acc = accuracy_score(y_test, y_hat)
    recall = recall_score(y_test, y_hat)
    print("Accuracy: ",acc,"Recall: " ,recall)
    conf_matrix=confusion_matrix(y_test,y_hat)
    print("confusion matrix :",conf_matrix)
    #---------Plotting--------
    ns_probablity = [0 for i in range(len(y_test))]
    print(ns_probablity)
    lr_probablity = g.predict_proba(X_test)
    lr_probablity = lr_probablity[:, 1]
    ns_fprobablity, ns_tprobablity, _ = roc_curve(y_test, ns_probablity,pos_label=2)
    lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probablity,pos_label=2)
    plt.plot(lr_fpr, lr_tpr, marker='.', label='Naive Bayes')
    plt.title("ROC Curve")
    plt.xlabel('False +ve Rate')
    plt.ylabel('True +ve Rate')
    plt.legend()
    plt.show()

read_data()
GaussNB()