import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
import os
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
plt.rcParams['figure.figsize'] = [15, 10]

# FUNCTION: Create pandas dataframe from numpy array
def createPandasDataFrame(X,Y,X_names,Y_names):
  df_X = pd.DataFrame(data=X, columns =X_names)
  df_Y = pd.DataFrame(data=Y, columns =Y_names)
  return df_X, df_Y

# CLASS: Easily import different datasets
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
      self.dataset_scelto = "diabetes" # Regressione default choice
    
    # Print dataset information
    #self.printDatasetInformation()

  def printDatasetInformation(self):
    #print(dataset_scelto)
    parameters = self.dataset_scelto.keys()
    data = self.dataset_scelto.values()
    #print(parameters)
    # Print useful information
    for name in parameters:
      print("------------------------------------------")
      print(name , self.dataset_scelto[name])
      print("------------------------------------------")

  def getXY(self):
    # Get Input (X) Data
    X = self.dataset_scelto['data'] # or  data = iris.get('data')
    X_names = self.dataset_scelto['feature_names']
    
    # Get Output (Y) Target
    parameters = self.dataset_scelto.keys()
    Y = self.dataset_scelto['target']
    Y_names = self.dataset_scelto['target_names']
    
    return X,Y,X_names,Y_names

# CLASS: Logistic Regression    
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

if __name__ == "__main__":

    # Load sklearn dataset
    # Regressione: "boston", "diabetes",
    # Classificazione: "iris", "digits", "wine", "breast_cancer "
    # Regressione: "diabetes", "boston", "linnerud"
    dataset_name = "iris"
    myScikitLearnDatasets=ScikitLearnDatasets(dataset_name)
    X,Y,X_names,Y_names = myScikitLearnDatasets.getXY()

    # https://www.datatechnotes.com/2019/05/one-hot-encoding-example-in-python.html
    from sklearn.preprocessing import OneHotEncoder
    onehot_encoder=OneHotEncoder(sparse=False)
    reshaped=Y.reshape(len(Y), 1)
    Y_onehot=onehot_encoder.fit_transform(reshaped)

    df_X,df_Y = createPandasDataFrame(X,Y,X_names,["Output"])
    
    # Split data into train and test
    X_train, X_test, Y_train, Y_test = train_test_split(df_X.values, df_Y.values, test_size=0.20, random_state=42)

    print(Y_train.shape)
    # Apply Logistic regression 
    myModel = LogisticRegression()
    trained_model = myModel.train(X_train, Y_train.ravel())
    Y_pred,score = myModel.evaluate(X_test,Y_test.ravel(),trained_model)
    print("Score Logistic regression: ",score)
    myModel.plot(Y_test,Y_pred)

