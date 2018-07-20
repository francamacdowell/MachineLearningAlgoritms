#Classification using Nearest Neighbors -------

#Example: Classifying Breast Cancer Samples -----

import pandas as pd
from sklearn import preprocessing

#Importing the data with pandas:
wisc_bc_data = pd.read_csv("wisc_bc_data.csv", sep=",")

#To see better the features of the data frame:
#wisc_bc_data.info()

#The ID column is useless for us, we can drop it:
wisc_bc_data = wisc_bc_data.drop(columns=['id'])

#The variable we want to predict is the diagnoses (If it's Benign or Malignant).
#We can check how many for each kind with this:
wisc_bc_data.diagnosis.value_counts()

#To work with Machine Learning classifiers, we need our data as categorical type:
wisc_bc_data['diagnosis'] = wisc_bc_data['diagnosis'].astype('category')

#And we can make our labes 'b' and 'm' more informative:
wisc_bc_data['diagnosis'] = wisc_bc_data['diagnosis'].cat.rename_categories(['Benign', 'Malignant'], inplace=False)

#To see the percentage proportion, we pass normalize parameter:
wisc_bc_data.diagnosis.value_counts(normalize=True)

#Analyzing the  feature values, we see that they are in different ranges ---
#We have to normalize them ---

#We can't normalize our labels category, so I'm separating the data
data_to_norm = wisc_bc_data.drop(columns = ['diagnosis'])
category_labels = wisc_bc_data['diagnosis']

#Saving columns' names
normalized_columns = data_to_norm.columns

#Instacing the normalizer
min_max_scaler = preprocessing.MinMaxScaler()
#Normalizing the data, the return of this transform, is a numpy array 
norm_bc_array = min_max_scaler.fit_transform(data_to_norm[:])
#Reshapening to a DataFrame
norm_bc_data = pd.DataFrame(norm_bc_array.reshape(len(data_to_norm.index),-1), columns=normalized_columns)

#Here we have another way to normalize our data
norm_bc_array2 = preprocessing.normalize(data_to_norm[:], norm='l2')
norm_bc_data2 = pd.DataFrame(norm_bc_array2.reshape(len(data_to_norm.index),-1), columns=normalized_columns)

##But we are going to use the first data normalized

#To check if the normalization worked, we can look to the area_mean description of data:
wisc_bc_data['area_mean'].describe()
norm_bc_data['area_mean'].describe()

#Now I'm going to create the training and test datasets:
wisc_bc_train = norm_bc_data[:468] #469 instances to train
wisc_bc_test = norm_bc_data[469:] #100 instances to test

#I also have to separate our labels in train and test to evaluate our model:
wisc_bc_train_labels = category_labels[:468]
wisc_bc_test_labels = category_labels[469:]

#It's time to create kNN classifier!
from sklearn.neighbors import KNeighborsClassifier
#Intancing our classifier with number K of neighbours:
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(wisc_bc_train, wisc_bc_train_labels) #fitting our model

accuracy = knn.score(wisc_bc_test, wisc_bc_test_labels)
print("Test set score:" + str(accuracy))

#If you want to know the labels predicts you can predict():
print(knn.predict(wisc_bc_test))
