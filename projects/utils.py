from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
import os
from sklearn import linear_model

def standardScaler(df):
    # Input pandas dataframe Output pandas dataframe scaled
    names = list(df.keys())
    data = df.values
    data_scaled = StandardScaler().fit_transform(data)
    df_scaled = pd.DataFrame(data = data_scaled , columns=names)
    print(df)
    print(df_scaled)
    return df_scaled

class dimensionalityReduction():

    def __init__(self,n_components): 
        # define pca
        self.pca = PCA(n_components)
        
    def create_names(self,col_number):
        names = []
        for i in range(0,col_number): 
            name = "component_" + str(i)
            names.append(name)
        return names

    def total_variance(self):
        # Varianza associata ad ogni componente
        variance_arr = self.pca.explained_variance_ratio_
        tot_variance = 0
        for variance in variance_arr:
            temp_variance = variance*100
            tot_variance += temp_variance
        return tot_variance

    def fit(self,df):
        df_scaled = standardScaler(df)
        model = self.pca.fit(df_scaled.values)
        return model

    def transform(self, model,df):  
        # Input dataframe
        principalComponents = model.transform(df.values)
        names = self.create_names(col_number=principalComponents.shape[1])
        df_scaled = pd.DataFrame(data = principalComponents , columns = names)
        print(df_scaled)
        tot_variance = self.total_variance()
        print(tot_variance)
        return df_scaled

class ScikitLearnDatasets():
  
  def __init__(self, dataset_name):
    # Load all scikit-learn dataset
    if ("iris"==dataset_name):
      self.dataset_scelto = datasets.load_iris() # Classificazione iris dataset
    elif ("digits"==dataset_name):
      self.dataset_scelto = datasets.load_digits() # Classificazione Load digits dataset
    elif ("wine"==dataset_name):
      self.dataset_scelto = datasets.load_wine() # Classificazione Load wine dataset
    elif ("breast_cancer"==dataset_name):
      self.dataset_scelto = datasets.load_breast_cancer() # Classificazione Load breast_cancer dataset
    elif ("boston"==dataset_name):
      self.dataset_scelto = datasets.load_boston() # Regressione Load boston dataset
      self.dataset_scelto.update([ ('target_names', ['Boston-House-Price'])] )
    elif ("diabetes"==dataset_name):
      self.dataset_scelto = datasets.load_diabetes() # Regressione Load diabetes dataset
      self.dataset_scelto.update([ ('target_names', ['Desease-Progression'])] )
    elif ("linnerud"==dataset_name):
      self.dataset_scelto = datasets.load_linnerud() # Regressione Load linnerud dataset
    else:
      self.dataset_scelto = diabetes # Regressione default choice
    
    # Print dataset information
    self.printDatasetInformation()

  def printDatasetInformation(self):
    #print(dataset_scelto)
    parametri = self.dataset_scelto.keys()
    valore = self.dataset_scelto.values()
    print(parametri)
    # Print useful information
    for name in parametri:
      print("------------------------------------------")
      print(name , self.dataset_scelto[name])
      print("------------------------------------------")

  def getXY(self):
    # Get Input (X) Data
    X = self.dataset_scelto['data'] # or  data = iris.get('data')
    X_names = self.dataset_scelto['feature_names']
    
    # Get Output (Y) Target
    parametri = self.dataset_scelto.keys()
    Y = self.dataset_scelto['target']
    Y_names = self.dataset_scelto['target_names']
    
    print("Dataset Parameters: ", parametri)
    print("Feature Names: ", X_names)
    print("Output Names: ", Y_names)
    print("Input X Shape: " , X.shape)
    print("Output Y Shape: " , Y.shape)
    
    return X,Y,X_names,Y_names
          
  def createPandasDataFrame(self,X,Y,X_names,Y_names):
    df_X = pd.DataFrame(data=X, columns =X_names)
    df_Y = pd.DataFrame(data=Y, columns =Y_names)
    return df_X, df_Y

  def writeDataFrameToCsv(self,df_X,df_Y,dataset_name):
    directory = "ScikitLearnDataset"
    if not os.path.exists(directory):
      os.makedirs(directory)
    
    # Create csv file
    path_write = os.path.join(directory, dataset_name)
    df_X.to_csv(path_write + '_X.csv', sep = ',', index = False)
    df_Y.to_csv(path_write + '_Y.csv', sep = ',', index = False)
    
class LinearRegression(): 

    def __init__(self):
      # Inizializzazione
      self.model = linear_model.LinearRegression(fit_intercept=True, normalize=False)
    
    def train(self,X,Y):
      # Stimare w0, w1 .. wN
      trained_model = self.model.fit(X,Y)
      print("w1,w2 .. wN : ",self.model.coef_)
      print("w0 : ", self.model.intercept_) 
      return trained_model
    
    def predict(self,X_test,trained_model):
      Y_pred = trained_model.predict(X_test)
      return Y_pred
    
    def evaluate(self,X_test, Y_test, trained_model):
      # R2 score
      print(X_test.shape)
      Y_pred = trained_model.predict(X_test)
      score = trained_model.score(X_test, Y_test)
      return Y_pred,score

    def plot(self,Y_test,Y_pred):
        length = Y_pred.shape[0] # 20
        index_bar = np.linspace(0,length,length)
        plt.plot(index_bar, Y_test, label='Test')
        plt.plot(index_bar, Y_pred, label='Prediction')
        plt.legend()
        plt.show()

class LogisticRegression():
  
    def __init__(self):
      # Inizializzazione
      # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
      self.model = linear_model.LogisticRegression(random_state=0,solver="lbfgs", multi_class="auto")
    
    def train(self,X,Y):
      trained_model = self.model.fit(X,Y)
      return trained_model

    def predict(self,X_test,trained_model):
      #Y_pred = trained_model.predict(X_test) # Predict label
      Y_pred = trained_model.predict_proba(X_test) # Predict probabilities
      return Y_pred

    def evaluate(self,X_test, Y_test, trained_model):
      # Mean accuracy
      Y_pred = trained_model.predict(X_test)
      score = trained_model.score(X_test, Y_test)
      return Y_pred, score

    def plot(self,Y_test,Y_pred):
        length = Y_pred.shape[0] # 20
        index_bar = np.linspace(0,length,length)
        plt.plot(index_bar, Y_test, label='Test')
        plt.plot(index_bar, Y_pred, label='Prediction')
        plt.legend()
        plt.show()