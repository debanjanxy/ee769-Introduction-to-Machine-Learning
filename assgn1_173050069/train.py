#!/usr/bin/python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
import math 
from sklearn import ensemble
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score
import pickle
from sklearn.tree import DecisionTreeClassifier


#load the housing training and testing data
def load_housing_data():
    return pd.read_csv("train.csv")


#get all the attributes which has some missing value
def get_error_attributes(housing_data):
    res = []
    feature_names = housing_data.columns
    X = housing_data.isna().any() # it contains info about the presence of null values in any column
    for f in feature_names:
        if X[f]:
            res.append(f)
    return res


#do one hot encoding of all text all attributes
def encode_text_attributes(text_attributes,housing_data):
    encoder = preprocessing.LabelEncoder()
    t_a = text_attributes.columns[0]
    for attr in t_a:
        housing_cat_encoded = encoder.fit_transform(housing_data)
        housing_data[attr] = housing_cat_encoded
    return housing_data


#split training data into two parts train and test data for validation
def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data)*test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


#load whole data in a dictionary, and load the mean mode of the error attributes in another dictionary
def convert_to_dict(error_features, all_features,housing_data):
	dcn = {}
	dcn_stat = {}
	for e in all_features:
	    y1 = list(housing_data[e])
	    dcn[e] = y1
	for e in error_features:
	    y = housing_data[e]
	    tf = y.isnull()
	    if(e=='MasVnrArea'):
	        M = y.mean()
	        dcn_stat[e] = M
	    else:
	        M = y.mode()
	        dcn_stat[e] = M	
	for e in error_features:
	    a = dcn_stat[e]
	    if a.dtype == np.object:
	        a1 = str(a[0])
	        dcn_stat[e] = a1
	    else:
	        a1 = float(a)
	        dcn_stat[e] = a1
	return dcn, dcn_stat    


#remove all the NaN values from the whole data dictionary
def remove_null_values(error_features, dcn, dcn_stat):
	for e in error_features:
		q = dcn[e]
		dcn[e] = [str(x) for x in q]
		for i in range(len(q)):
			if dcn[e][i]=='nan':
				dcn[e][i] = dcn_stat[e]
	dcn['GarageYrBlt'] = list(map(float,dcn['GarageYrBlt']))
	dcn['GarageYrBlt'] = list(map(int,dcn['GarageYrBlt']))
	dcn['MasVnrArea'] = list(map(float,dcn['MasVnrArea']))
	housing = pd.DataFrame(dcn)
	return housing	


# One Hot Encoding of Categorical Values
def do_one_hot_encode(housing):
	categorical_attr = housing.select_dtypes(include=['object'])
	le = preprocessing.LabelEncoder()
	ca = categorical_attr.apply(le.fit_transform)
	housing_t = housing		# housing_t # one hot encoded whole feature data set
	data_Y = ca['SaleStatus']	# data_Y # label encoded whole target data set
	data_Y2 = housing['SaleStatus']		# data_Y2 # categorical target values(soldfast=1, soldslow=2, notsold=3)
	for i in categorical_attr.columns:
	    x = pd.get_dummies(housing_t[i])
	    if i=='SaleStatus':
	        data_Y1 = x 	# data_Y1 # one hot encoded whole target data set
	    else:
	        housing_t = pd.concat([housing_t,x],axis=1)
	    del housing_t[i]
	whole_housing_data = pd.concat([housing_t,data_Y],axis=1) # clean and clear data set # whole_housing_data.to_csv('hello.csv',sep='\t')
	return whole_housing_data


#here we are plotting the graph between accuracy and gamma
def _plot_svm_gamma(svm_train_X, svm_test_X, train_Y, test_Y):
	gamma_lst = [0.0000001, 0.000005, 0.00005, 0.0001, 0.0002, 0.0003, 0.0004,0.0005, 0.0006, 0.0007, 0.0008,
	0.0009, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.05, 0.1, 1, 10]
	acc_lst = []
	best_gamma = 0.0
	best_acc = 0.00
	c = 80
	for i in gamma_lst:
		svm1 = svm.SVC(gamma=i)
		svm1.fit(svm_train_X, train_Y)
		pred = svm1.predict(svm_test_X)
		a = accuracy_score(test_Y, pred)
		acc_lst.append(a)
		if a>best_acc:
			best_acc = a
			best_gamma = i
	plt.figure('SVM (Accuracy Vs. Gamma)')
	plt.plot(gamma_lst, acc_lst,'--bo')
	plt.xscale('log')
	plt.ylabel('Accuracy')
	plt.xlabel('Gamma')
	plt.show()
	return best_gamma


#here we are plotting the graph between accuracy and c
def _plot_svm_c(svm_train_X, svm_test_X, train_Y, test_Y, best_gamma):
	c_lst = [0.01, 0.1, 1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 
	300, 400, 500, 1000, 2000, 5000, 10000, 100000, 1000000]
	acc_lst = []
	best_c = 0.0
	best_acc = 0.00
	g = best_gamma
	for i in c_lst:
		svm1 = svm.SVC(C=i,gamma=g)
		svm1.fit(svm_train_X, train_Y)
		pred = svm1.predict(svm_test_X)
		a = accuracy_score(test_Y, pred)
		acc_lst.append(a)
		if a>best_acc:
			best_acc = a
			best_c = i
	plt.figure('SVM (Accuracy Vs. C)')
	plt.plot(c_lst, acc_lst,'--bo')
	plt.xscale('log')
	plt.ylabel('Accuracy')
	plt.xlabel('C')
	plt.show()
	return best_c

	
#classification using svm and selecting the best parameters
def svm_classifier(train_X, train_Y, test_X, test_Y):
    import matplotlib.pyplot as plt    
    train_X1 = preprocessing.normalize(train_X)
    svm_train_X = preprocessing.scale(train_X1)
    test_X1 = preprocessing.normalize(test_X)
    svm_test_X = preprocessing.scale(test_X1)
    best_gamma = _plot_svm_gamma(svm_train_X, svm_test_X, train_Y, test_Y)
    best_c = _plot_svm_c(svm_train_X, svm_test_X, train_Y, test_Y, best_gamma)
    clf_svm = svm.SVC(C=best_c,gamma=best_gamma)
    clf_svm.fit(svm_train_X,train_Y)
    pred = clf_svm.predict(svm_test_X)
    print("--------------------------------------------------------------")
    print("SVM Best C Value: ",best_c)
    print("SVM Best Gamma Value: ",best_gamma)
    print("SVM Classifer Accuracy: ",accuracy_score(test_Y,pred))
    svm_pkl_filename = 'model0.pkl'
    svm_pkl_model = open(svm_pkl_filename,'wb')
    pickle.dump(clf_svm, svm_pkl_model)
    svm_pkl_model.close()


#plot between learning rate and accuracy in gradient boosting
def _plot_gb_learning_rate(train_X, train_Y, test_X, test_Y):
	lr = [0.00001, 0.001, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.02, 0.03, 0.04, 
	0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 1, 5, 10]
	acc_lst = []
	best_lr = 0.0
	best_acc = 0.00
	for i in lr:
		clf_grd_boost = ensemble.GradientBoostingClassifier(learning_rate=i)
		clf_grd_boost.fit(train_X,train_Y)
		pred = clf_grd_boost.predict(test_X)
		a = accuracy_score(test_Y, pred)
		acc_lst.append(a)
		if a>best_acc:
			best_acc = a
			best_lr = i
	plt.figure('Gradient Boosting (Accuracy Vs. Learning Rate)')
	plt.plot(lr, acc_lst,'--bo')
	plt.xscale('log')
	plt.ylabel('Accuracy')
	plt.xlabel('Learning Rate')
	plt.show()
	return best_lr


#plot between n_estimators and accuracy in gradient boosting
def _plot_gb_n_estimators(train_X, train_Y, test_X, test_Y, best_learning_rate):
	ne_lst = [1, 5, 10, 30, 50, 80, 100, 200, 300, 500]
	acc_lst = []
	best_ne = 0
	best_acc = 0.00
	for i in ne_lst:
		clf_grd_boost = ensemble.GradientBoostingClassifier(n_estimators=i, learning_rate=best_learning_rate)
		clf_grd_boost.fit(train_X,train_Y)
		pred = clf_grd_boost.predict(test_X)
		a = accuracy_score(test_Y, pred)
		acc_lst.append(a)
		if a>best_acc:
			best_acc = a
			best_ne = i
	plt.figure('Gradient Boosting (Accuracy Vs. N_Estimators)')
	plt.plot(ne_lst, acc_lst,'--bo')
	plt.xscale('log')
	plt.ylabel('Accuracy')
	plt.xlabel('N Estimators')
	plt.show()
	return best_ne


#Gradient Boosting Classifier
def gradient_boosting_classifier(train_X, train_Y, test_X, test_Y):	
	best_learning_rate = _plot_gb_learning_rate(train_X, train_Y, test_X, test_Y)
	best_n_estimators = _plot_gb_n_estimators(train_X, train_Y, test_X, test_Y, best_learning_rate)
	print("--------------------------------------------------------------")
	print("Gradient Boosting Best Learning Rate: ", best_learning_rate)
	print("Gradent Boosting Best N Estimator: ", best_n_estimators)
	
	#comment
	clf_grd_boost = ensemble.GradientBoostingClassifier(learning_rate = best_learning_rate, n_estimators=best_n_estimators)
	clf_grd_boost.fit(train_X,train_Y)
	pred = clf_grd_boost.predict(test_X)
	print("Gradient Boosting Classifier Accuracy: ",accuracy_score(test_Y,pred))
	print("--------------------------------------------------------------")
	gb_pkl_filename = 'model1.pkl'
	gb_pkl_model = open(gb_pkl_filename,'wb')
	pickle.dump(clf_grd_boost, gb_pkl_model)
	gb_pkl_model.close()


#plot between learning rate and accuracy in Adaboosting
def _plot_ab_learning_rate(train_X, train_Y, test_X, test_Y):
	lr = [0.00001, 0.001, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.02, 0.03, 0.04, 
	0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 1, 5, 10]
	acc_lst = []
	best_lr = 0.0
	best_acc = 0.00
	for i in lr:
		clf_grd_boost = ensemble.AdaBoostClassifier(DecisionTreeClassifier(max_depth=10), learning_rate=i)
		clf_grd_boost.fit(train_X,train_Y)
		pred = clf_grd_boost.predict(test_X)
		a = accuracy_score(test_Y, pred)
		acc_lst.append(a)
		if a>best_acc:
			best_acc = a
			best_lr = i
	plt.figure('AdaBoosting (Accuracy Vs. Learning Rate)')
	plt.plot(lr, acc_lst,'--bo')
	plt.xscale('log')
	plt.ylabel('Accuracy')
	plt.xlabel('Learning Rate')
	plt.show()
	return best_lr


#plot between n_estimators and accuracy in Adaboosting
def _plot_ab_n_estimators(train_X, train_Y, test_X, test_Y, best_learning_rate):
	ne_lst = [1, 5, 10, 30, 50, 80, 100, 200, 300, 500]
	acc_lst = []
	best_ne = 0
	best_acc = 0.00
	for i in ne_lst:
		clf_grd_boost = ensemble.AdaBoostClassifier(DecisionTreeClassifier(max_depth=10), n_estimators=i, learning_rate=best_learning_rate)
		clf_grd_boost.fit(train_X,train_Y)
		pred = clf_grd_boost.predict(test_X)
		a = accuracy_score(test_Y, pred)
		acc_lst.append(a)
		if a>best_acc:
			best_acc = a
			best_ne = i
	plt.figure('AdaBoosting (Accuracy Vs. N_Estimators)')
	plt.plot(ne_lst, acc_lst,'--bo')
	plt.xscale('log')
	plt.ylabel('Accuracy')
	plt.xlabel('N Estimators')
	plt.show()
	return best_ne


#AdaBoost Classifier
def adaboost_classifier(train_X,train_Y,test_X,test_Y):
	best_learning_rate = _plot_ab_learning_rate(train_X, train_Y, test_X, test_Y)
	best_n_estimators = _plot_ab_n_estimators(train_X, train_Y, test_X, test_Y, best_learning_rate)
	print("--------------------------------------------------------------")
	print("AdaBoosting Best Learning Rate: ", best_learning_rate)
	print("AdaBoosting Best N Estimator: ", best_n_estimators)
	clf_softmax = ensemble.AdaBoostClassifier(DecisionTreeClassifier(max_depth=10), learning_rate=best_learning_rate, n_estimators=best_n_estimators)
	clf_softmax.fit(train_X,train_Y)
	pred = clf_softmax.predict(test_X)
	print("AdaBoost Classifier Accuracy: ", accuracy_score(test_Y,pred))
	print("--------------------------------------------------------------")
	sftmax_pkl_filename = 'model2.pkl'
	sftmax_pkl_model = open(sftmax_pkl_filename,'wb')
	pickle.dump(clf_softmax,sftmax_pkl_model)
	sftmax_pkl_model.close()


#plot between n_estimators and accuracy in Random Forest
def _plot_rf_n_estimators(train_X, train_Y, test_X, test_Y):
	ne_lst = [1, 5, 10, 15, 20, 25, 30, 35, 40, 50, 60, 70, 80, 90, 100]
	acc_lst = []
	best_ne = 0
	best_acc = 0.00
	for i in ne_lst:
		clf_grd_boost = ensemble.RandomForestClassifier(n_estimators=i)
		clf_grd_boost.fit(train_X,train_Y)
		pred = clf_grd_boost.predict(test_X)
		a = accuracy_score(test_Y, pred)
		acc_lst.append(a)
		if a>best_acc:
			best_acc = a
			best_ne = i
	plt.figure('Random Forest (Accuracy Vs. N_Estimators)')
	plt.plot(ne_lst, acc_lst,'--bo')
	plt.xscale('log')
	plt.ylabel('Accuracy')
	plt.xlabel('N Estimators')
	plt.show()
	return best_ne


##plot between max_features and accuracy in random forest
def _plot_rf_max_features(train_X, train_Y, test_X, test_Y):
	ne_lst = [1, 5, 10, 15, 20, 25, 30, 35, 40, 50, 60, 70, 80, 90, 100, 106]
	acc_lst = []
	best_ne = 0
	best_acc = 0.00
	for i in ne_lst:
		clf_grd_boost = ensemble.RandomForestClassifier(max_features=i)
		clf_grd_boost.fit(train_X,train_Y)
		pred = clf_grd_boost.predict(test_X)
		a = accuracy_score(test_Y, pred)
		acc_lst.append(a)
		if a>best_acc:
			best_acc = a
			best_ne = i
	plt.figure('Random Forest (Accuracy Vs. Max Features)')
	plt.plot(ne_lst, acc_lst,'--bo')
	plt.xscale('log')
	plt.ylabel('Accuracy')
	plt.xlabel('Max Features')
	plt.show()
	return best_ne


#Random Forest Classifier
def random_forest_classifier(train_X,train_Y,test_X,test_Y):
	best_n_estimators = _plot_rf_n_estimators(train_X, train_Y, test_X, test_Y)
	best_max_features = _plot_rf_max_features(train_X, train_Y, test_X, test_Y)
	clf_rfc = ensemble.RandomForestClassifier(n_estimators=best_n_estimators, max_features=best_max_features)
	clf_rfc.fit(train_X,train_Y)
	pred = clf_rfc.predict(test_X)
	print("--------------------------------------------------------------")
	print("Random Forest Best N Estimator: ", best_n_estimators)
	print("Random Forest Best Max_Features: ", best_max_features)
	print("Random Forest Classifier accuracy: ", accuracy_score(test_Y,pred))
	print("--------------------------------------------------------------")
	rfc_pkl_filename = 'model3.pkl'
	rfc_model_pkl = open(rfc_pkl_filename,'wb')
	pickle.dump(clf_rfc, rfc_model_pkl)
	rfc_model_pkl.close()

#preprocess the data by removing NaN values and encoding text values
def preprocess_data():
    housing_data = load_housing_data()	#load the data
    all_features = housing_data.columns	#get all the feature names
    error_features = get_error_attributes(housing_data)	#get all the feature names containing NaN values
    
    dcn,dcn_stat = convert_to_dict(error_features,all_features,housing_data)	#convert the dataframe to dictionary for simplicity of removal of null values
    
    housing = remove_null_values(error_features, dcn, dcn_stat)	#removal of null values and convert the dictionary to dataframe
    
    whole_housing_data = do_one_hot_encode(housing)	#do one hot encoding on the dataframe 
    # splitting of the data has been done
    train_set, test_set = split_train_test(whole_housing_data, 0.2) # split the data into test and train set
    train_X = np.array(train_set) # convert train_X to a numpy array
    train_X = train_set.iloc[:,0:len(train_set.iloc[0])-1] # training feature matrix
    train_Y = train_set['SaleStatus'] # training label vector
    test_X = test_set.iloc[:,0:len(train_set.iloc[0])-1] # testing feature matrixcategorical_attr = housing_data.select_dtypes(include=['object'])
    test_Y = test_set['SaleStatus'] # testing label vector
    return train_X,train_Y,test_X,test_Y


if __name__=='__main__':
    
    train_X,train_Y,test_X,test_Y = preprocess_data()	# preprocess the data

    svm_classifier(train_X, train_Y, test_X, test_Y)	#prediction using svm classifier
    
    gradient_boosting_classifier(train_X,train_Y,test_X,test_Y)		#prediction using gradient boosting
    
    adaboost_classifier(train_X,train_Y,test_X,test_Y)		#prediction using softmax classifier
    
    random_forest_classifier(train_X,train_Y,test_X,test_Y)		#prediction using random forest classifier

