from utils import *
from sklearn.model_selection import train_test_split

if __name__ == "__main__":

    # Load sklearn dataset
    # Regressione: "boston", "diabetes",
    # Classificazione: "iris", "digits", "wine", "breast_cancer "
    # Regressione: "diabetes", "boston", "linnerud"
    dataset_name = "diabetes"
    myScikitLearnDatasets=ScikitLearnDatasets(dataset_name)
    X,Y,X_names,Y_names = myScikitLearnDatasets.getXY()
    df_X,df_Y = myScikitLearnDatasets.createPandasDataFrame(X,Y,X_names,Y_names)
    myScikitLearnDatasets.writeDataFrameToCsv(df_X,df_Y,dataset_name)

    # Split data into train and test
    X_train, X_test, Y_train, Y_test = train_test_split(df_X.values, df_Y.values, test_size=0.20, random_state=42)

    # Dimensionality reduction Input dataframe output dataframe
    df_X_train = pd.DataFrame(data = X_train , columns=X_names)
    df_X_test = pd.DataFrame(data = X_test , columns=X_names)
    n_components = 0.95 # 90% of the variance
    mydimensionalityReduction = dimensionalityReduction(n_components)
    pcaModel = mydimensionalityReduction.fit(df_X_train)
    df_X_train_scaled = mydimensionalityReduction.transform(pcaModel,df_X_train)
    df_X_test_scaled = mydimensionalityReduction.transform(pcaModel,df_X_test)
    X_train_scaled = df_X_train_scaled.values
    X_test_scaled = df_X_test_scaled.values
    print("Number of inputs with PCA: ",X_train_scaled.shape[1])
    print("Number of inputs without PCA: ",X_test_scaled.shape[1])
    
    # Apply Linear regression with PCA
    myModelPCA = LinearRegression()
    trained_modelPCA = myModelPCA.train(X_train_scaled, Y_train)
    Y_predPCA,scorePCA = myModelPCA.evaluate(X_test_scaled,Y_test,trained_modelPCA)
    print("Score Linear Regression PCA: ", scorePCA)
    #myModelPCA.plot(Y_test,Y_pred)

    # Apply Linear regression 
    myModel = LinearRegression()
    trained_model = myModel.train(X_train, Y_train)
    Y_pred,score = myModel.evaluate(X_test,Y_test,trained_model)
    print("Score Linear regression without PCA: ",score)
    #myModel.plot(Y_test,Y_pred)

    # Comparison
    length = Y_pred.shape[0] # 20
    index_bar = np.linspace(0,length,length)
    plt.plot(index_bar, Y_test, label='Test')
    plt.plot(index_bar, Y_predPCA, label='PredictionPCA')
    plt.plot(index_bar, Y_pred, label='Prediction')
    plt.legend()
    plt.show()


# Cose da fare e domande
'''
OK: Quando scalo devo scalare anche l'output? SI transform sempre
Regressione come trattare i booleani?
Usa una regresisone diversa
read from csv sklearn datasets
aggiungi logistic regression example
ad correlation plot and others
'''