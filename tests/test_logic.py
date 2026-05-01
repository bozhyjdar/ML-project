import pytest
from src.data_loader import load_xor, load_subscribers

def test_xor_data_shape():
    """Test sprawdzający poprawność wymiarów danych XOR"""
    X, y = load_xor()
    assert X.shape == (4, 2)
    assert len(y) == 4

def test_subscribers_split():
    """Test sprawdzający podział zbioru Subscribers na część treningową i testową"""
    X_train, X_test, y_train, y_test = load_subscribers()
    assert len(X_train) > 0
    assert len(X_test) > 0
    assert len(X_train) + len(X_test) == 1000

def test_xor_logic():
    """Test weryfikujący logikę bramki XOR w danych wejściowych"""
    X, y = load_xor()
    assert y[1] == 1  # 0 XOR 1 = 1
    assert y[3] == 0  # 1 XOR 1 = 0