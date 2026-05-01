import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from tensorflow.keras.optimizers import Adam, SGD


from src.data_loader import (load_subscribers, load_xor, load_titanic, 
                             load_mnist, load_breast_cancer_data)
from src.models import (build_xor_model, build_titanic_model, 
                        build_mnist_cnn, build_cancer_mlp)


os.makedirs('plots', exist_ok=True)

def run_experiments():

    print("--- 1. Zbiór Subscribers ---")
    X_train, X_test, y_train, y_test = load_subscribers()
    
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    print("Naiwny Bayes Dokładność:", accuracy_score(y_test, nb.predict(X_test)))
    
    dt = DecisionTreeClassifier(max_depth=5)
    dt.fit(X_train, y_train)
    print("Drzewo Decyzyjne Dokładność:", accuracy_score(y_test, dt.predict(X_test)))

   
    print("\n--- 2. Sieć MLP: Problem XOR ---")
    X_xor, y_xor = load_xor()
    model_xor = build_xor_model()
    model_xor.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.1), metrics=['accuracy'])
    history_xor = model_xor.fit(X_xor, y_xor, epochs=200, verbose=0)
    
    plt.plot(history_xor.history['loss'])
    plt.title('Błąd uczenia - Bramka XOR')
    plt.xlabel('Epoka')
    plt.ylabel('Loss (Strata)')
    plt.savefig('plots/xor_loss.png')
    plt.clf()

  
    print("\n--- 3. Sieć MLP: Zbiór Titanic ---")
    X_train_t, X_test_t, y_train_t, y_test_t = load_titanic()
    model_t = build_titanic_model()
    model_t.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    history_t = model_t.fit(X_train_t, y_train_t, epochs=50, validation_data=(X_test_t, y_test_t), verbose=0)
    
    plt.plot(history_t.history['loss'], label='Strata Treningowa')
    plt.plot(history_t.history['val_loss'], label='Strata Walidacyjna')
    plt.title('Błąd uczenia - Titanic')
    plt.xlabel('Epoka')
    plt.ylabel('Loss (Strata)')
    plt.legend()
    plt.savefig('plots/titanic_loss.png')
    plt.clf()
    print("Dokładność Titanic (Zbiór testowy):", model_t.evaluate(X_test_t, y_test_t, verbose=0)[1])


    print("\n--- 4. Sieć CNN: Zbiór MNIST ---")
    (x_train_m, y_train_m), (x_test_m, y_test_m) = load_mnist()
    model_cnn = build_mnist_cnn()
    model_cnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model_cnn.fit(x_train_m, y_train_m, epochs=3, validation_data=(x_test_m, y_test_m))
    print("Dokładność MNIST (Zbiór testowy):", model_cnn.evaluate(x_test_m, y_test_m, verbose=0)[1])

   
    print("\n--- 5. Wymagania na ocenę 5.0 (Breast Cancer Wisconsin) ---")
    X_c, y_c, X_train_c, X_test_c, y_train_c, y_test_c = load_breast_cancer_data()
    
   
    subset = X_c.columns[:5]
    sns.pairplot(pd.concat([X_c[subset], pd.Series(y_c, name='target')], axis=1), hue='target')
    plt.savefig('plots/eda_cancer.png')
    plt.clf()
    print("Zapisano wykresy EDA.")


    dt_cancer = DecisionTreeClassifier(max_depth=4)
    dt_cancer.fit(X_train_c, y_train_c)
    print("Dokładność Drzewa (Breast Cancer):", accuracy_score(y_test_c, dt_cancer.predict(X_test_c)))

    
    input_dim = X_train_c.shape[1]
    
   
    model_mom = build_cancer_mlp(input_dim)
    model_mom.compile(loss='binary_crossentropy', optimizer=SGD(learning_rate=0.01, momentum=0.9), metrics=['accuracy'])
    hist_mom = model_mom.fit(X_train_c, y_train_c, epochs=50, validation_data=(X_test_c, y_test_c), verbose=0)
    
  
    model_adam = build_cancer_mlp(input_dim)
    model_adam.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])
    hist_adam = model_adam.fit(X_train_c, y_train_c, epochs=50, validation_data=(X_test_c, y_test_c), verbose=0)
    
   
    plt.plot(hist_mom.history['val_loss'], label='SGD + Momentum (Walidacja)')
    plt.plot(hist_adam.history['val_loss'], label='Adam (Walidacja)')
    plt.title('Porównanie metod uczenia - Breast Cancer')
    plt.xlabel('Epoka')
    plt.ylabel('Loss (Strata)')
    plt.legend()
    plt.savefig('plots/advanced_learning.png')
    plt.clf()
    
    print("\nKoniec eksperymentów! Wszystkie wykresy zapisano w folderze 'plots/'.")

if __name__ == "__main__":
    run_experiments()