import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.linear_model as sk
df = pd.DataFrame()
learn_rate=0.01
iteration=1000

def read_data():
    """
    Used for the Reading the data
    Parameters-None

    """
    global df
    # reading data seperated by ;
    df = pd.read_csv('diabetes2.csv', sep=',')
    #print(df.head)
def preprocess():
    """
    Used for the Pre-Processing the data
    Parameters-None
    Returns- X,y

    """
    read_data()
    np.random.seed(0)
    df['Pregnancies'].replace(to_replace=0, value=df['Pregnancies'].median(),inplace=True)
    df['Glucose'].replace(to_replace=0, value=df['Glucose'].median(),inplace=True)
    df['BloodPressure'].replace(to_replace=0, value=df['BloodPressure'].median(),inplace=True)
    df['SkinThickness'].replace(to_replace=0, value=df['SkinThickness'].median(),inplace=True)
    df['Insulin'].replace(to_replace=0, value=df['Insulin'].median(),inplace=True)
    df['BMI'].replace(to_replace=0, value=df['BMI'].median(),inplace=True)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    #print(X)
    X=(X-X.mean(axis=0))
    X=X/X.std(axis=0)
    return X,y
def sigmoid(z):
    """
    Used for the calculation of 1/1+e^-x 
    Parameters-z
    Returns- sigmoid of z

    """
    return 1.0/(1 + np.exp(-z))
def cost(X, y, th_0):
    """
    Used for the Calculation of Loss 
    Parameters-None
    Returns- Training and Testing Data

    """
    m=y.shape[0]
    temp_sigm=sigmoid(X.dot(th_0))
    check_0=(temp_sigm==0)
    temp_sigm[check_0]=1e-10
    check_1=(temp_sigm==1)
    temp_sigm[check_1]-=1e-10
    #print(X.dot(theta))
    #X^t (sigmoid(X*theta)-y)
    theta_dash=(X.T).dot(temp_sigm-y)
    cal_error=(-(y) * np.log(temp_sigm) - (1 - y) * np.log(1 - temp_sigm))
    return theta_dash,cal_error.mean()

def predict( X, theta):
    """
    Used for the Predicting 
    Parameters-X, Theta 
    Returns- Training and Testing Data
    """
    X = np.hstack((np.ones((X.shape[0], 1)), X))
    y_hat = sigmoid(X.dot(theta))
    y_temp=y_hat >= 0.5
    y_hat[y_temp] = 1
    y_temp=y_hat < 0.5
    y_hat[y_temp] = 0
    # return the numpy array y which contains the predicted values
    return y_hat
def accuracy(pre, org):
    """
    Used for the Calculating the Accuracy 
    Parameters-predicted value ,original value
    Returns- Accuracy

    """
    TP=0
    TN=0
    FP=0
    FN=0
    for i in range(len(pre)):
        if(pre[i]==1 and org[i]==1):
            TP+=1
        elif(pre[i]==0 and org[i]==1):
            FN+=1
        elif(pre[i]==1 and org[i]==0):
            FP+=1
        elif(pre[i]==0 and org[i]==0):
            TN+=1

    acc = (TP+TN)/(TP+TN+FP+FN)
    return acc
def zip_cal(y,y_hat):
    """
    Simple function used just for the calculation 
    Parameters-predicted value ,original value
    Returns- False Positive, False Negative, True Positive, True Negative

    """
    FP = 0
    FN = 0
    TP = 0
    TN = 0
    for y, y_pred in zip(y, y_hat):
            if y_pred == y:
                if y_pred == 1:
                    TP += 1
                else:
                    TN += 1  
            else:
                if y_pred == 1:
                    FP += 1
                else:
                    FN += 1
    return FP,FN,TP,TN
def Calculate( y, y_hat):
    """
    Used for the Calculating the Accuracy 
    Parameters-predicted value ,original value
    Returns- Accuracy

    """
    m = len(y)
    fp,fn,tp,tn=zip_cal(y,y_hat)
    Confusion_matrix = np.array([[tn, fp], [fn, tp]])
    acc = accuracy(y,y_hat)
    precision = np.divide(tp, (tp+fp))
    recall = np.divide(tp, (tp+fn))
    f1 = 2*(precision * recall)/(precision+recall)
    return Confusion_matrix, acc, precision, recall, f1
def BGD(X,y,X_val,y_val):
    """
    Used for the Calculation of Batch Gradient Descent 
    Parameters-training data, val data
    Returns- theta, training cost, val cost

    """
    global iteration
    n=X.shape[0]
    X = np.hstack((np.ones((n, 1)), X))
    th_0=np.zeros((X.shape[1],))
    val_cost=[]
    train_cost=[]
    X_val=np.hstack((np.ones((X_val.shape[0],1)),X_val))
    for i in range(iteration+1):
        m=len(X)
        random_train=np.random.randint(m)-1
        random_Xtrain=[]
        random_Xtrain.append(X[random_train])
        numpy_Xtrain=np.array(random_Xtrain)
        random_ytrain=[]     
        random_ytrain.append(y[random_ytrain])     
        numpy_ytrain=np.array(random_ytrain)
        train_th0dash,train_loss=cost(numpy_Xtrain,numpy_ytrain,th_0)
        val_th0dash,val_loss=cost(X_val,y_val,th_0)
        train_cost.append(train_loss)
        val_cost.append(val_loss)
        th_0=th_0-(learn_rate*train_th0dash)
    return th_0,train_cost,val_cost
    
def SGD(X,y,X_val,y_val):
    """
    Used for the Calculation of  Stochastic gradient descent  
    Parameters-training data, val data
    Returns- theta, training cost, val cost

    """
    global iteration
    n=X.shape[0]
    X = np.hstack((np.ones((n, 1)), X))
    th_0=np.zeros((X.shape[1],))
    val_cost=[]
    train_cost=[]
    X_val=np.hstack((np.ones((X_val.shape[0],1)),X_val))
    for i in range(iteration+1):
        train_th0dash,train_loss=cost(X,y,th_0)
        val_th0dash,val_loss=cost(X_val,y_val,th_0)
        train_cost.append(train_loss)
        val_cost.append(val_th0dash)
        th_0=th_0-(learn_rate*train_th0dash)
    return th_0,train_cost,val_cost

def plot_graph(training_loss, validation_loss, gradient):
    """
    Used for the Plotting the Graph
    Parameters-training loss, validation loss, 
    Returns- theta, training cost, val cost

    """
    plt.plot(training_loss,label="%s Train Loss at α=%.4g" % (gradient, learn_rate))
    plt.plot(validation_loss,label="%s Val Loss at α=%.4g" % (gradient,learn_rate))
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
def comparing_alphas(X_train,y_train,X_val,y_val):
    """
    Used for the Comparing Alphas  
    Parameters-training data, validation data
    Returns- None

    """
    alpha=[0.0005, 0.01, 0.0001, 0.005,10]
    for i in alpha:
        learning_rate = alpha
        train_theta_BGD, train_loss_BGD, validate_loss_BGD = BGD(X_train, y_train, X_val, y_val)
        # print(train_theta_BGD)
        train_theta_SGD, train_loss_SGD, validate_loss_SGD = SGD(X_train, y_train, X_val, y_val)
        # print(train_theta_SGD)
        plot_graph(train_loss_BGD, validate_loss_BGD,'BGD')
        plot_graph(train_loss_SGD, validate_loss_SGD, 'SGD')

def logistech_regression():
    """
    Used for the Driver function   
    Parameters-None
    Returns- None

    """
    X,y=preprocess()
    n = X.shape[0]
    X_train = X[:int(n*0.7)]
    Y_train = y[:int(n*0.7)]

    
    valueofval=n-int(n*0.7)-int(((n-int(n*0.7))*2)/3)
    valueoftest=n-int(n*0.7)-valueofval
    X_value=X[int(n*0.7):-valueoftest]
    y_value=y[int(n*0.7):-valueoftest]

    X_test=X[-valueoftest:]
    y_test=y[-valueoftest:]
    #print(value0fval," ",valueoftest)
    #Y_pred=predict(X_train,th_0)

    #th_0_train_BGD,train_loss_BGD,val_loss_BGD=BGD(X_train,Y_train,X_value,y_value)
    th_0_train_SGD,train_loss_SGD,val_loss_SGD=SGD(X_train,Y_train,X_value,y_value)


    #-----------------Comparing alphas-------
    #comparing_alphas(X_train,Y_train,X_value,y_value)

    #-----BGD------
    # y_dash=predict(X_test,th_0_train_BGD)
    # CM,acc,precison,recall,f1=Calculate(y_test,y_dash)
    # print("Confusion Matrix BGD : ",CM)
    # print("Accuracy BGD : ",acc)
    # print("Precisiom BGD : ",precison)
    # print("Recall BGD : ",recall)
    # print("F1 BGD : ",f1)

    #----SGD--------
    # y_dash=predict(X_test,th_0_train_SGD)
    # CM,acc,precison,recall,f1=Calculate(y_test,y_dash)
    # print("Confusion Matrix SGD : ",CM)
    # print("Accuracy SGD : ",acc)
    # print("Precisiom SGD : ",precison)
    # print("Recall SGD : ",recall)
    # print("F1 SGD : ",f1)


    #--------------2------------
    logitech=sk.LogisticRegression(max_iter=1000)
    logitech.fit(X_train,Y_train)
    y_pred=logitech.predict(X_test)
    CM,acc,precison,recall,f1=Calculate(y_test,y_pred)
    print("Confusion Matrix : ",CM)
    print("Accuracy : ",acc)
    print("Precisiom : ",precison)
    print("Recall : ",recall)
    print("F1 : ",f1)

logistech_regression()
