import joblib
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

def train_and_eval(
    test_size=0.2,
    random_state=1012,
    model_path=None,
):
    columnas = ['source_port', 'destination_port', 'protocol', 'packets', 'length','fin_flag', 'syn_flag', 'rst_flag','psh_flag', 'ack_flag', 'urg_flag','cwe_flag', 'ece_flag', 'Label']
    df_shuffle = pd.read_csv("./df_5000_final_14_caracteristicas.csv", usecols = columnas, skipinitialspace=True)
    features = df_shuffle.drop("Label", axis=1).values
    labels = df_shuffle["Label"].values
    
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_size, random_state=random_state)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model_dnn_1 = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(64, 128, 256, 256, 256, 128, 128, 64, 8), random_state=1, activation='relu')
    model_dnn_1.fit(X_train, y_train)

    y_pred = model_dnn_1.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    recall = metrics.recall_score(y_test, y_pred, average='weighted')
    f1 = metrics.f1_score(y_test, y_pred, average='weighted')
    precision = sklearn.metrics.precision_score(y_test, y_pred, labels=None, pos_label=1, average='macro', sample_weight=None, zero_division='warn')
    results = {
        'accuracy': accuracy,
        'recall': recall,
        'f1': f1,
        'precision' : precision
    }
    if model_path:
        joblib.dump(model_dnn_1, model_path)
    return results
