from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
import os
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
plt.rcParams['figure.figsize'] = [15, 10]

# FUNCTION: Standard data with 0 mean and unit variance (Gaussian)
def standardScaler(df):
    # Input pandas dataframe Output pandas dataframe scaled
    names = list(df.keys())
    data = df.values
    data_scaled = StandardScaler().fit_transform(data)
    df_scaled = pd.DataFrame(data = data_scaled , columns=names)
    #print(df)
    #print(df_scaled)
    return df_scaled

# FUNCTION: Create pandas dataframe from numpy array
def createPandasDataFrame(X,Y,X_names,Y_names):
  df_X = pd.DataFrame(data=X, columns =X_names)
  df_Y = pd.DataFrame(data=Y, columns =Y_names)
  return df_X, df_Y

# FUNCTION: Write pandas dataframe to csv file
def writeDataFrameToCsv(df_X,df_Y,directory_path,dataset_name):
  if not os.path.exists(directory_path):
    os.makedirs(directory_path)
  
  # Create csv file
  path_write = os.path.join(directory_path, dataset_name)
  df_X.to_csv(path_write + '_X.csv', sep = ',', index = False)
  df_Y.to_csv(path_write + '_Y.csv', sep = ',', index = False)

# FUNCTION: Read from csv file a dataset
def readDataFrameFromCsv(directory_path, dataset_name):
  path_read = os.path.join(directory_path, dataset_name)
  if not os.path.exists(directory_path):
    print("Directory path does not exist")
  df_X = pd.read_csv(path_read + '_X.csv') 
  df_Y = pd.read_csv(path_read + '_Y.csv') 
  return df_X, df_Y  

# CLASS: PCA class to do input dimensionality reduction
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

    def fit(self,df):
        df_scaled = standardScaler(df)
        model = self.pca.fit(df_scaled.values)
        information_array = model.explained_variance_ratio_ *100.00
        total_information = np.sum(information_array)
        return model,information_array, total_information

    def transform(self, model,df):  
        # Input dataframe
        principalComponents = model.transform(df.values)
        names = self.create_names(col_number=principalComponents.shape[1])
        df_scaled = pd.DataFrame(data = principalComponents , columns = names)
        return df_scaled

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
          
# CLASS: Linar Regression          
class LinearRegression(): 

    def __init__(self):
      # Inizializzazione
      # https://scikit-learn.org/stable/modules/linear_model.html#ridge-regression-and-classification
      self.model = linear_model.LinearRegression(fit_intercept=True, normalize=False)
      
    def train(self,X,Y):
      # Stimare w0, w1 .. wN
      trained_model = self.model.fit(X,Y)
      #print("w1,w2 .. wN : ",self.model.coef_)
      #print("w0 : ", self.model.intercept_) 
      return trained_model
    
    def predict(self,X_test,trained_model):
      Y_pred = trained_model.predict(X_test)
      return Y_pred
    
    def evaluate(self,X_test, Y_test, trained_model):
      # R2 score
      Y_pred = trained_model.predict(X_test)
      R2_score = trained_model.score(X_test, Y_test)
      RMSE_score = (np.sqrt(mean_squared_error(Y_test, Y_pred)))
      return Y_pred,R2_score, RMSE_score

    def plot(self,Y_test,Y_pred):
        length = Y_pred.shape[0] # 20
        index_bar = np.linspace(0,length,length)
        plt.plot(index_bar, Y_test, label='Test')
        plt.plot(index_bar, Y_pred, label='Prediction')
        plt.legend()
        plt.show()


if __name__ == "__main__":

    # ----------- SKLEARN DATASET LOADING ------------ #
    # Load sklearn dataset
    # Classificazione: "iris", "digits", "wine", "breast_cancer "
    # Regressione: "diabetes", "boston", "linnerud"
    # 1. Select dataset 
    dataset_name = "diabetes"
    # 2. Create class object ScikitLearnDatasets 
    myScikitLearnDatasets=ScikitLearnDatasets(dataset_name)
    # 3. Print dataset information
    #myScikitLearnDatasets.printDatasetInformation()
    # 4. Get dataset data as numpy array X=input, Y=output and X_names=input_names, Y_names=output_names
    X,Y,X_names,Y_names = myScikitLearnDatasets.getXY()
    # 5. Convert numpy array data to Pandas Dataframe
    df_X,df_Y = createPandasDataFrame(X,Y,X_names,Y_names)
    print("#---------- DATASET INFORMATION ------------#")
    print("X Input or feature_names: ", X_names)
    print("Y Output or target_names: ", Y_names)
    print("Input X Shape: " , X.shape)
    print("Output Y Shape: " , Y.shape)
    print("Dataframe df_X Input Describe: \n", df_X.describe())
    print("Dataframe df_Y Output Describe: \n", df_Y.describe())
    print("#-------------------------------------------#")
    # 6. Write Pandas dataframe df_X, df_Y to csv file
    directory_path = os.path.join(os.getcwd(), "ScikitLearnDatasets")
    writeDataFrameToCsv(df_X,df_Y,directory_path, dataset_name)
    # ------------------------------------------------ #

    # ----------- READ DATASET FROM CSV -------------- #
    # Read previously saved dataset
    # 1. Read csv dataset (examvle boston_X.csv and boston_Y.csv) and transform to pandas daframe
    #dataset_name = "boston" # desired dataset name 
    #directory_path = os.path.join(os.getcwd(), "ScikitLearnDatasets") # dataset folder
    df_X,df_Y = readDataFrameFromCsv(directory_path, dataset_name)
    # ------------------------------------------------ #

    # -------- Split data into train and test -------- #
    # Split dataset into training and test set
    X_train, X_test, Y_train, Y_test = train_test_split(df_X.values, df_Y.values, test_size=0.20, random_state=42)
    # ------------------------------------------------ #

    # -------------------- PCA ----------------------- #
    # Principal component analysis (PCA) or dimensionality reduction
    # Number of input is reduced while keeping overall dataset information
    # 1. Covert numpy array to pandas dataframe
    df_X_train = pd.DataFrame(data = X_train , columns=df_X.keys())
    df_X_test = pd.DataFrame(data = X_test , columns=df_X.keys())
    # 2. Initialize PCA (Principal Component Analysis)
    n_components = 0.95 # 90% of the variance
    mydimensionalityReduction = dimensionalityReduction(n_components)
    # 3. Create PCA model (using input training data)
    pcaModel,information_array, total_information = mydimensionalityReduction.fit(df_X_train)
    print("#-------------- PCA ANALYSIS ---------------#")
    print("Information for each new component: ", information_array, "%")
    print("Total Information of the reduced dataset: ", total_information, " %")
    # 4. Apply created PCA model to both training and test dataset
    df_X_train_scaled = mydimensionalityReduction.transform(pcaModel,df_X_train)
    df_X_test_scaled = mydimensionalityReduction.transform(pcaModel,df_X_test)
    X_train_scaled = df_X_train_scaled.values
    X_test_scaled = df_X_test_scaled.values
    #print("Dataset X_train: ", X_train)
    #print("Dataset X_train Reduced: ", X_train_scaled)
    print("Number of inputs with PCA: ",X_train_scaled.shape[1])
    print("Number of inputs without PCA: ",X_train.shape[1])
    print("#-------------------------------------------#")
    # --------------------------------------------------#
    
    # -------- LINEAR REGRESSION WITH PCA DATA -------- #
    # we use reduce input data
    myModelPCA = LinearRegression()
    trained_modelPCA = myModelPCA.train(X_train_scaled, Y_train)
    Y_predPCA,R2_scorePCA, RMSE_scorePCA = myModelPCA.evaluate(X_test_scaled,Y_test,trained_modelPCA)
    print("#----- LINEAR REGRESSION PCA RESULTS -------#")
    print("w1,w2 .. wN : ",trained_modelPCA.coef_)
    print("w0 : ", trained_modelPCA.intercept_) 
    print("Score Linear Regression PCA: ", "R2 Score: ", R2_scorePCA, " RMSE Score: ", RMSE_scorePCA)
    print("#-------------------------------------------#")
    #myModelPCA.plot(Y_test,Y_pred)
    # ------------------------------------------------ #

    # -------------- LINEAR REGRESSOR ---------------- #
    # We use initial data 
    myModel = LinearRegression()
    trained_model = myModel.train(X_train, Y_train)
    Y_pred,R2_score, RMSE_score = myModel.evaluate(X_test,Y_test,trained_model)
    print("#------- LINEAR REGRESSION RESULTS ---------#")
    print("w1,w2 .. wN : ",trained_modelPCA.coef_)
    print("w0 : ", trained_modelPCA.intercept_) 
    print("Score Linear regression without PCA: ", "R2 Score: ", R2_score, " RMSE Score: ", RMSE_score)
    print("#-------------------------------------------#")
    #myModel.plot(Y_test,Y_pred)
    # ------------------------------------------------ #

    #----------------- COMPARISON -------------------- #    
    length = Y_pred.shape[0] # 20
    index_bar = np.linspace(0,length,length)
    plt.plot(index_bar, Y_test, label='Test')
    plt.plot(index_bar, Y_predPCA, label='PredictionPCA')
    plt.plot(index_bar, Y_pred, label='Prediction')
    plt.legend()
    plt.show()
    # ------------------------------------------------ #
