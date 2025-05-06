from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

def load_data():
    digits = load_digits()
    X = digits.images.reshape((len(digits.images), -1))  # flatten 8x8 to 64
    y = digits.target
    return train_test_split(X, y, test_size=0.2, random_state=42)
