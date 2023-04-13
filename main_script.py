import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import (
    train_test_split,
    KFold,
    StratifiedKFold,
    cross_val_score
)
from sklearn.metrics import classification_report
import tensorflow as tf
import joblib

DATA_PATH = '/home/armak/Python_projects_WSL/Forest_cover_type_classification/covtype.data'

# Implementing a simple heuristic model
class Heuristic:

    """A simple heuristic model that always predicts the most common class.
        args:
            X: feature matrix
            y: target vector
    """

    def __init__(self, X, y):
        """Initializes the model"""
        self.X = X
        self.y = y

    def predict(self, x):
        """Predicts the most common class"""
        return self.y.value_counts().index[0]
    
    def evaluate(self, X, y):
        """Calculates the accuracy of the model"""
        return np.mean(y == self.predict(X))

def load_data(path, val_data=False):

    """Loads the data and splits it into train, validation and test sets and scales the data with Z-score standardization.
        args:
            path: path to the data
            val_data: whether to split the data into validation and test sets
        returns:
            X_train_scaled: scaled train set
            X_test_scaled: scaled test set
            X_val_scaled: scaled validation set
            y_train: train set labels
            y_test: test set labels
            y_val: validation set labels
    """

    cont_names = [
    "Elevation",
    "Aspect",
    "Slope",
    "R_Hydrology",
    "Z_Hydrology",
    "R_Roadways",
    "Hillshade_9am",
    "Hillshade_Noon",
    "Hillshade_3pm",
    "R_Fire_Points",
    ] # Continuous variables

    area_names = ['WArea_' + str(i + 1) for i in range(4)]
    soil_names = ['Soil_' + str(i + 1) for i in range(40)]
    cat_names = area_names + soil_names # Categorical variables
    target = 'Cover_Type'
    names = cont_names + cat_names # All column names except target

    df = pd.read_csv(path, header=None, names=names+[target])
    X, y = df.iloc[:,:-1], df.iloc[:,-1]
    y = y - 1 # Change the target values to start from 0
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=0)
    sc = StandardScaler().fit(X_train) 
    X_train_scaled = sc.transform(X_train)
    if val_data:
        X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, shuffle=True, random_state=1)
        X_val_scaled = sc.transform(X_val)
    else:
        X_val_scaled = None
        y_val = None
    X_test_scaled = sc.transform(X_test)
    return X_train_scaled, X_test_scaled, X_val_scaled, y_train, y_test,  y_val

# Logistic regression training 
def train_logistic_regression_model(X_train, y_train):

    """Trains a logistic regression model and saves it to working directory.
        args:
            X_train: training feature matrix
            y_train: training target vector"""
    
    log_model = LogisticRegression(max_iter=1000)
    log_model.fit(X_train,y_train)
    joblib.dump(log_model, 'log_model.pkl')
    return log_model

# KNN training
def train_knn_model(X_train, y_train, n_neighbors=5):

    """Trains a KNN model and saves it to working directory.
        args:
            X_train: training feature matrix
            y_train: training target vector
            n_neighbors: number of neighbors"""
    
    knn_model = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn_model.fit(X_train,y_train)
    joblib.dump(knn_model, 'knn_model.pkl')
    return knn_model

# Evaluating the models
def evaluate_model(model_path, X_test, y_test, cv=None):

    """Loads the model and evaluates it on the test set.
        args:
            model_path: path to the model
            X_test: test feature matrix
            y_test: test target vector
            cv: number of folds for cross validation score
        returns:
            classification_report: classification report
            cross_val_score: cross validation score
    """
    
    model = joblib.load(model_path)
    return classification_report(y_test,model.predict(X_test)), cross_val_score(model, X_test, y_test, cv=cv)

# A simple neural network model
def train_nn_model(X_train, y_train, X_val, y_val, num_nodes, dropout_prob, learning_rate, batch_size, epochs):

    """Trains a neural network model and returns the model and the history object.  
        args:
            X_train: training feature matrix
            y_train: training target vector
            X_val: validation feature matrix
            y_val: validation target vector
            num_nodes: number of nodes in each hidden layer
            dropout_prob: dropout probability
            learning_rate: learning rate
            batch_size: batch size
            epochs: number of epochs
        returns:
            nn_model: the trained neural network model
            history: the history object
    """
    
    nn_model = tf.keras.Sequential([
        tf.keras.layers.Dense(num_nodes,activation='relu', input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dropout(dropout_prob),
        tf.keras.layers.Dense(num_nodes,activation='relu'),
        tf.keras.layers.Dropout(dropout_prob),
        tf.keras.layers.Dense(len(y_train.unique()),activation='softmax')])
    nn_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = nn_model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val))
    return nn_model, history

def plot_history (history):

    """Plots the training and validation loss and accuracy.
        args:
            history: history object returned by model.fit()"""
    
    fig, (ax1,ax2) = plt.subplots(1,2, figsize=(10,4))
    ax1.plot(history.history['loss'], label='loss')
    ax1.plot(history.history['val_loss'], label='val_loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True)
    ax1.legend()

    ax2.plot(history.history['accuracy'], label='accuracy')
    ax2.plot(history.history['val_accuracy'], label='val_accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.grid(True)
    ax2.legend()

    plt.show()

# Hyperparameter tuning function
def hyperparameter_tuning(model, X_train, y_train, X_val, y_val):

    """Tunes the hyperparameters of a neural network model.
        args:
            model: the neural network model
            X_train: training feature matrix
            y_train: training target vector
            X_val: validation feature matrix
            y_val: validation target vector"""
    
    least_val_loss = float('inf')
    least_loss_model = None
    least_loss_history = None
    epochs = 100
    for num_nodes in [16, 32, 64]:
        for dropout_prob in [0, 0.2]:
            for lr in [0.005, 0.001]:
                for batch_size in [64, 128]:
                    print(f'{num_nodes} nodes, dropout {dropout_prob}, learning_rate {lr}, batch_size {batch_size}, ')
                    model, history = train_nn_model(X_train, 
                                                    y_train, 
                                                    X_val, 
                                                    y_val,
                                                    num_nodes, 
                                                    dropout_prob, 
                                                    lr,
                                                    batch_size, 
                                                    epochs) 
                    val_loss, _ = model.evaluate(X_val, y_val)
                    if  val_loss < least_val_loss:
                        least_val_loss = val_loss
                        least_loss_model = model
                        least_loss_history = history
    return least_loss_model, least_loss_history


if __name__ == '__main__':

    # Load the data
    X_train, X_test, X_val, y_train, y_test, y_val = load_data(DATA_PATH, val_data=True)

    # Train the models
    heuristic_model = Heuristic(X_train, y_train)
    log_model = train_logistic_regression_model(X_train, y_train)
    knn_model = train_knn_model(X_train, y_train)
    nn_model, history = train_nn_model(X_train, y_train, X_val, y_val, 128, 0.2, 0.001, 128, 10)
    plot_history(history)
    # nn_model, history = hyperparameter_tuning(nn_model, X_train, y_train, X_val, y_val)
    # nn_model = tf.keras.models.load_model('/home/armak/Python_projects_WSL/Forest_cover_type_classification/nn_model.h5')

    nn_model.save('nn_model.h5')

    # Evaluate the models
    print(heuristic_model.evaluate(X_test, y_test)) # Heuristic model evaluation
    print(evaluate_model('log_model.pkl', X_test, y_test)) # Logistic regression model evaluation
    print(evaluate_model('knn_model.pkl', X_test, y_test)) # KNN model evaluation
    nn_model.evaluate(X_test, y_test) # Neural network model evaluation