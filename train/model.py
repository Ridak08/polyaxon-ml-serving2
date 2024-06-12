import joblib
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense,GRU,Embedding,Dropout,Flatten,Conv1D,MaxPooling1D,LSTM
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier

def train_and_eval(
    test_size=0.2,
    random_state=1012,
    model_path=None,
):
    columnas = ['source_port', 'destination_port', 'protocol', 'packets', 'length','fin_flag', 'syn_flag', 'rst_flag','psh_flag', 'ack_flag', 'urg_flag','cwe_flag', 'ece_flag', 'Label']
    df_shuffle = pd.read_csv("./df_5000_final_14_caracteristicas.csv", usecols = columnas, skipinitialspace=True)
    features = train_features_scaled.shape[1]
    nClasses = len(df[' Label'].unique())

    # Importación de bibliotecas y División de datos
    train_df, test_df = train_test_split(df_shuffle, test_size = 0.20) #Division del dataframe en un conjunto de  entrenamiento y prueba con un 85% de entrenamiento y 15% de prueba.
    
    # Separación de características y etiquetas 
    train_features = train_df.copy()
    train_labels = train_features.pop(' Label')
    
    test_features = test_df.copy()
    test_labels = test_features.pop(' Label')
    
    # Creación de un diccionario de características
    train_features_dict = {name: np.array(value) 
                             for name, value in train_features.items()}
    
    # Estandarización de características (llevar a la misma escala)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    train_features_scaled = scaler.fit_transform(train_features)
    
    test_features_scaled = scaler.transform(test_features)
    train_features_scaled.shape

    keras.backend.clear_session()
    tf.random.set_seed(42)
    np.random.seed(42)
    
    model_cnn_2 = keras.models.Sequential([
        keras.layers.Conv1D(32, kernel_size=4, strides=2, padding='same', activation='relu', input_shape=(features, 1)),
        keras.layers.Conv1D(64, kernel_size=4, strides=2, padding='same', activation='relu'),
        keras.layers.MaxPooling1D(),
        keras.layers.Conv1D(128, kernel_size=2, strides=2),  # Reduced kernel size to 2
        keras.layers.Flatten(),
        keras.layers.Dense(80, activation='relu'),
        keras.layers.Dense(45, activation='relu'),
        keras.layers.Dense(nClasses, activation='softmax')
    ])
    num_epochs = 25

    model_cnn_2.compile(optimizer='adam', loss= 'sparse_categorical_crossentropy', metrics= ['accuracy'])
    early_stopping_cb = keras.callbacks.EarlyStopping(patience = 10, restore_best_weights = True)

    model_cnn_2.fit(train_features_scaled, train_labels, epochs = num_epochs, callbacks = [early_stopping_cb], validation_split= 0.20)
    
    c = model_cnn_2.predict(test_features_scaled)
    a=np.argmax(c, axis=1)
    b = test_labels

    acc = (accuracy_score(a, b)*100)
    prec = sklearn.metrics.precision_score(b, a, labels=None, pos_label=1, average='macro', sample_weight=None, zero_division='warn')
    rec = metrics.recall_score(b, a, labels=None, pos_label=1, average='weighted', sample_weight=None)
    f1 = f1_score(a, b, average='macro')
    results = results._append({'Method':'CNN 2', 'Accuracy':acc, 'Precision':prec, 'F1_Score':f1, 'Recall':rec}, ignore_index=True)

    if model_path:
        joblib.dump(model_dnn_1, model_path)
    return results
