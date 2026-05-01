from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten

def build_xor_model():
    model = Sequential([
        Dense(8, input_dim=2, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    return model

def build_titanic_model():
    model = Sequential([
        Dense(16, input_dim=4, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    return model

def build_mnist_cnn():
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')
    ])
    return model

def build_cancer_mlp(input_dim):
    model = Sequential([
        Dense(32, input_dim=input_dim, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    return model