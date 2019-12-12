from utils import *
from sklearn.model_selection import train_test_split

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

    df_X,df_Y = myScikitLearnDatasets.createPandasDataFrame(X,Y,X_names,["Output"])
    myScikitLearnDatasets.writeDataFrameToCsv(df_X,df_Y,dataset_name)

    # Split data into train and test
    X_train, X_test, Y_train, Y_test = train_test_split(df_X.values, df_Y.values, test_size=0.70, random_state=42)

    print(Y_train.shape)
    # Apply Logistic regression 
    myModel = LogisticRegression()
    trained_model = myModel.train(X_train, Y_train.ravel())
    Y_pred,score = myModel.evaluate(X_test,Y_test.ravel(),trained_model)
    print("Score Logistic regression: ",score)
    myModel.plot(Y_test,Y_pred)


# Cose da fare e domande
'''
OK: Quando scalo devo scalare anche l'output? SI transform sempre
Regressione come trattare i booleani?
Usa una regresisone diversa
read from csv sklearn datasets
aggiungi logistic regression example
ad correlation plot and others
'''