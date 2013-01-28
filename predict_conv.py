from sklearn import preprocessing,ensemble,svm,cross_validation
from sklearn.metrics import mean_squared_error,explained_variance_score,mean_absolute_error,r2_score
import numpy as np
import scipy
import matplotlib.pyplot as plt
from sklearn.svm import SVR

def main():
    #keyword_id clicks conv cost date group hour imps match_type month monthday pos weekday
    data= np.genfromtxt('data.csv', delimiter=',', dtype=None)[1:]

    target = data[:,2].astype(np.float) # conv
    data = scipy.delete(data,2,1)

    skf = cross_validation.StratifiedKFold(target,n_folds=10)
    for train_index, test_index in skf:
        X_train, X_test = data[train_index], data[test_index]
        y_train, y_test = target[train_index], target[test_index]

        X_train,X_test = preprocess(X_train,X_test)
        train_data =np.column_stack( [ X_train , y_train ] )
        train_data = np.array([np.array(x) for x in set(tuple(x) for x in train_data)])
        X_train = train_data[:,:-1]
        y_train = train_data[:,-1]
        #predicted_test = random_forest_regressor(X_train,y_train,X_test)
        predicted_test = svm(X_train,y_train,X_test)
        _,_,non_zero_clicks = get_non_zero_clicks(X_test,y_test)
        predicted_test[non_zero_clicks==False] = 0
        model_eval(y_test,predicted_test)

def preprocess(train_data,test_data):

    #### training data

    train_data = scipy.delete(train_data,3,1) # delete Date attribute
    train_data = scipy.delete(train_data,4,1) # delete hour
    train_data = scipy.delete(train_data,4,1) #delete imps

    #encoding
    encoder = {}
    #Encoding
    encoder['group'] = preprocessing.LabelEncoder()
    train_data[:,3] = encoder['group'].fit_transform(train_data[:,3]).astype(np.float) #group

    encoder['match_type'] = preprocessing.LabelEncoder()
    train_data[:,4] = encoder['match_type'].fit_transform(train_data[:,4]).astype(np.float) #match_type

    encoder['weekday'] = preprocessing.LabelEncoder()#weekday
    train_data[:,8]= encoder['weekday'].fit_transform(train_data[:,8]).astype(np.float)

    #Scaling
    scaler = {}
    scaler['cost'] = preprocessing.StandardScaler()
    train_data[:,2] = scaler['cost'].fit_transform(train_data[:,2].astype(np.float))



    #### Testing Data
    test_data = scipy.delete(test_data,3,1) # delete Date attribute
    test_data = scipy.delete(test_data,4,1) # delete hour
    test_data = scipy.delete(test_data,4,1) #delete imps

    test_data[:,3] = encoder['group'].transform(test_data[:,3]).astype(np.float)
    test_data[:,4] = encoder['match_type'].transform(test_data[:,4]).astype(np.float)
    test_data[:,8] = encoder['weekday'].transform(test_data[:,8]).astype(np.float)

    return train_data,test_data


def model_eval(target, predicted):  # model evaluation
    print "Fold:"
    print "explained_variance_score ", explained_variance_score(target, predicted)
    print "mean_absolute_error", mean_absolute_error(target, predicted)
    print "mean_squared_error", mean_squared_error(target, predicted)
    print "r2_score", r2_score(target, predicted)


def random_forest_regressor(train,train_target,test):
    train,target,_ = get_non_zero_clicks(train,train_target)

    random_forest = ensemble.RandomForestRegressor(n_estimators=10)

    return random_forest.fit(train,target).predict(test)


def svm(train,train_target,test):
    train,target,_ = get_non_zero_clicks(train,train_target)

    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
    svr_lin = SVR(kernel='linear', C=1e3)
    svr_poly = SVR(kernel='poly', C=1e3, degree=2)
    y_rbf = svr_rbf.fit(train, target).predict(test)
    #y_lin = svr_lin.fit(train, target).predict(test)
    #y_poly = svr_poly.fit(train, target).predict(test)
    return y_rbf

def get_non_zero_clicks(data,target=None):
    non_zero_clicks = data[:,1].astype(np.float)>1
    data = data[non_zero_clicks,:]
    if target is not None:
        target = target[non_zero_clicks]
    return data,target,non_zero_clicks

if __name__ == "__main__":
    main()