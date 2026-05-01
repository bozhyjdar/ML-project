import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml, load_breast_cancer, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf

def load_subscribers():
    
    X, y = make_classification(n_samples=1000, n_features=4, n_classes=2, random_state=42)
    return train_test_split(X, y, test_size=0.2)

def load_xor():
    
    X = np.array([[0,0], [0,1], [1,0], [1,1]])
    y = np.array([0, 1, 1, 0])
    return X, y

def load_titanic():
   
    titanic = fetch_openml('titanic', version=1, as_frame=True, parser='auto')
    df = titanic.data
    df['survived'] = titanic.target
    df = df[['pclass', 'sex', 'age', 'fare', 'survived']].dropna()
    df['sex'] = LabelEncoder().fit_transform(df['sex'])
    
    X = df.drop('survived', axis=1)
    y = df['survived'].astype(int)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return train_test_split(X_scaled, y, test_size=0.2, random_state=42)

def load_mnist():
    
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    return (x_train, y_train), (x_test, y_test)

def load_breast_cancer_data():
   
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X, y, X_train_scaled, X_test_scaled, y_train, y_test