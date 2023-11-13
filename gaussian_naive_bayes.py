import numpy as np
import random

random.seed(10)

# loading data into predictor matrix and desired value vector
def load_data(filename):
    # reading data
    data = np.loadtxt(filename,delimiter=',',dtype=float)
    # first 5 columns are predictors, last column in true value

    X = data[:,:-1] # predictor matrix
    Y = data[:,-1].reshape(data.shape[0],1) # true value vector
    
    return X,Y

def split_train_test(X,Y,p=0.8):
    
    m,n = X.shape
    
    c = int(m*p)
    
    indices = [i for i in range(m)]
    random.shuffle(indices)
    
    X_train = X[indices[:c],:]
    Y_train = Y[indices[:c]]
    
    X_test = X[indices[c:],:]
    Y_test = Y[indices[c:]]
    
    return X_train, Y_train, X_test, Y_test

def standardization(X_train,X):
    meanX = np.mean(X_train,axis=0)
    stdX = np.std(X_train,axis =0)
    X = (X - meanX)/stdX
    
    return X

def a_priori_probabilites(Y):
    phi = np.zeros(len(np.unique(Y)))
    for j in Y:
        phi[int(j)] += 1
    
    return phi/len(Y)

def mean_estimator(X,Y):
    m = len(np.unique(Y))
    n = X.shape[1]
    mean = np.zeros((m,n))
    for i in range(m):
        for j in range(n):
            mean[i,j] = np.mean(X[Y.reshape(Y.shape[0],)==i,j])
    
    return mean

def variance_estimator(X,Y,mean):
    m = len(np.unique(Y))
    n = X.shape[1]
    var = np.zeros((m,n))
    for i in range(m):
        for j in range(n):
            var[i,j] = np.mean((X[Y.reshape(Y.shape[0],)==i,j]-mean[i,j])**2)
    
    return var

def train_model(X,Y):
    phi = a_priori_probabilites(Y)
    mean = mean_estimator(X,Y)
    var = variance_estimator(X,Y,mean)
    
    return phi, mean, var

def gaussian(x,mean,var):
    return 1/np.sqrt(2*np.pi*var)*np.exp(-(x-mean)**2/(2*var))


def predict(X,phi,mean,var):
    probabilities = np.zeros((X.shape[0],len(phi)))
    for i in range(len(phi)):
        for j in range(X.shape[0]):
            probabilities[j,i] = phi[i]*np.prod(gaussian(X[j,:],mean[i,:],var[i,:]))
    
    return np.argmax(probabilities,axis=1).reshape((X.shape[0],1))

X, Y = load_data("multiclass_data.csv")

X_train, Y_train, X_test, Y_test = split_train_test(X,Y)

X_test = standardization(X_train, X_test)
X_train = standardization(X_train,X_train)

phi, mean, var = train_model(X_train,Y_train)

Y_pred_train = predict(X_train,phi,mean,var)
Y_pred_test = predict(X_test,phi,mean,var)

print("Accuracy on Train: " + str(np.sum(Y_pred_train == Y_train)/len(Y_train)*100)[:7] + " %")
print("Accuracy on Test: " + str(np.sum(Y_pred_test == Y_test)/len(Y_test)*100)[:7] + " %")


    