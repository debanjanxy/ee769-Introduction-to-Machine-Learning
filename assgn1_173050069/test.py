#!/usr/bin/python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
import math
import sklearn 
from sklearn import ensemble
from sklearn import svm
from sklearn.metrics import accuracy_score
import pickle


#load the housing training and testing data
def load_data(filename):
    return pd.read_csv(filename)


#get all the attributes which has some missing value
def get_error_attributes(housing_data):
    res = []
    feature_names = housing_data.columns
    X = housing_data.isna().any() # it contains info about the presence of null values in any column
    for f in feature_names:
        if X[f]:
            res.append(f)
    return res


#get all text/categorical attribute names
def get_all_text_attributes(housing_data):
    attributes = housing_data.columns
    result = []
    for attr in attributes:
        x = str(type(housing_data[attr][0]))
        if x=="<class 'str'>":
            result.append(attr)
    return result


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


#convert dataframe data to dictionary for error and noise removal
def convert_to_dict(error_features, all_features,housing_data):
	dcn = {}
	dcn_stat = {}
	for e in all_features:
	    y1 = list(housing_data[e])
	    dcn[e] = y1
	for e in error_features:
	    y = housing_data[e]
	    tf = y.isnull()
	    if(e=='MasVnrArea' or e=='GarageArea'):
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
    dcn['BsmtFullBath'] = list(map(float,dcn['BsmtFullBath']))
    dcn['BsmtHalfBath'] = list(map(float,dcn['BsmtHalfBath']))
    dcn['GarageCars'] = list(map(float,dcn['GarageCars']))
    dcn['GarageArea'] = list(map(float, dcn['GarageArea']))
    housing = pd.DataFrame(dcn)
    return housing


# One Hot Encoding of Categorical Values
def do_one_hot_encode(housing):
    categorical_attr = housing.select_dtypes(include=['object'])
    le = preprocessing.LabelEncoder()
    ca = categorical_attr.apply(le.fit_transform)
    housing_t = housing		# housing_t # one hot encoded whole feature data set
    for i in categorical_attr.columns:
        x = pd.get_dummies(housing_t[i])
        housing_t = pd.concat([housing_t,x],axis=1)
        del housing_t[i]
    return housing_t


#take test.csv and process it 
def preprocess_test_data(raw_data):
    test_data = load_data("test.csv")
    all_features = test_data.columns	#get all the feature names
    
    error_features = get_error_attributes(test_data)	#get all the feature names containing NaN values
    
    dcn,dcn_stat = convert_to_dict(error_features,all_features,test_data)	#convert the dataframe to dictionary for simplicity of removal of null values
    
    test = remove_null_values(error_features, dcn, dcn_stat)	#removal of null values and convert the dictionary to dataframe
    
    clean_data = do_one_hot_encode(test)	#do one hot encoding on the dataframe 
    return clean_data


#Label encode ground truth data
def prepare_gt_data(filename):
	data = load_data(filename)
	labels = data['Sold Fast'] #modification needed
	for i in range(len(labels)):
		print("Processing column ",i)
		if 'Fast' in str(labels.iloc[i]):
			labels.iloc[i] = 1
		elif 'Slow' in str(labels.iloc[i]):
			labels.iloc[i] = 2
		else:
			labels.iloc[i] = 3
	return labels 


#create out.csv and calculate accuracy
def write_to_csv(pred,Id):
	pred = list(pred)
	for i in range(len(pred)):
		if pred[i]==1:
			pred[i] = 'SoldFast'
		elif pred[i]==2:
			pred[i] = 'SoldSlow'
		else:
			pred[i] = 'NotSold'
	pred = pd.Series(pred,name='SoldStatus')
	data = load_data('gt.csv')
	labels = data.iloc[:,1:2]
	score = accuracy_score(pred, labels)
	df = pd.concat([Id,pred],axis=1)
	df.to_csv('out.csv',sep='\t',encoding='utf-8',index=False)
	return score


if __name__=="__main__":
	raw_data = load_data("test.csv")

	#clean test raw data
	clean_data = preprocess_test_data(raw_data)
	l = len(clean_data)
	clean_data['2.5Fin'] = pd.Series(np.zeros(l))
	RRAe = pd.Series(np.zeros(l),name='RRAe')
	RRAn = pd.Series(np.zeros(l),name='RRAn')
	RRNn = pd.Series(np.zeros(l),name='RRNn')
	missing = pd.concat([RRAe,RRAn,RRNn],axis=1)
	clean_data = pd.concat([clean_data,missing],axis=1)
	Id = clean_data['Id']
	
	#model creation and accuracy calculation
	try:
		gb_pkl_model = open('model1.pkl','rb')
		print("reading model1.pkl")
	except:
		print("Reading finalModel1.pkl")
		gb_pkl_model = open('finalModel1.pkl','rb')
	gb_model = pickle.load(gb_pkl_model)
	pred = gb_model.predict(clean_data)
	score = write_to_csv(pred,Id)
	print("=====================================")
	print("Final Accuracy: ", score)
	print("=====================================")

