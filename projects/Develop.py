from utils import *
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import MinMaxScaler,StandardScaler

if __name__ == "__main__":

    # ----------- SKLEARN DATASET LOADING ------------ #
    # Load sklearn dataset
    # Classificazione: "iris", "digits", "wine", "breast_cancer "
    # Regressione: "diabetes", "boston", "linnerud"
    # 1. Select dataset 
    dataset_name = "boston"
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

    # ---------------- SVM REGRESSOR ---------------- #
    # We use initial data 
    # Support Vector machine required data to be scaled
    myModelSVM = SVM()
    scaler = StandardScaler() #MinMaxScaler(feature_range=(0, 1))
    X_train_MinMax_scaled = scaler.fit_transform(X_train)
    X_test_MinMax_scaled = scaler.fit_transform(X_test)
    trained_modelSVM = myModelSVM.train(X_train_MinMax_scaled, Y_train)
    Y_predSVM,R2_scoreSVM, RMSE_scoreSVM = myModelSVM.evaluate(X_test_MinMax_scaled,Y_test,trained_modelSVM)
    print("#-------------- SVM RESULTS ----------------#")
    print("Score SVM regression without PCA: ", "R2 Score: ", R2_scoreSVM, " RMSE Score: ", RMSE_scoreSVM)
    print("#-------------------------------------------#")
    #myModel.plot(Y_test,Y_pred)
    # ------------------------------------------------ #

    #----------------- COMPARISON -------------------- #    
    length = Y_pred.shape[0] # 20
    index_bar = np.linspace(0,length,length)
    plt.plot(index_bar, Y_test, label='Test')
    plt.plot(index_bar, Y_predPCA, label='PredictionPCA')
    plt.plot(index_bar, Y_pred, label='Prediction')
    plt.plot(index_bar, Y_predSVM, label='PredictionSVM')
    plt.legend()
    plt.show()
    # ------------------------------------------------ #


# Cose da fare e domande
'''
Regressione come trattare i booleani?
Usa una regresisone diversa
ad correlation plot and others
add differnt metrics
'''