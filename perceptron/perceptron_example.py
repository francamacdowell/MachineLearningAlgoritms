import sklearn.linear_model
import pandas as pd
from sklearn.naive_bayes import GaussianNB

def main():
    sonar_data = pd.read_csv("sonar-data.csv", sep=',', header=None)
    a = sklearn.linear_model.Perceptron(max_iter=5)
    #print(len(sonar_data.columns))
    data_train = sonar_data.iloc[:, :59]
    label_train = sonar_data.iloc[:, 60]
    print(data_train.shape, label_train.shape)    

    a.fit(data_train, label_train,)
    #print(sonar_data.iloc[:, :60])
if __name__ == '__main__':
    main()