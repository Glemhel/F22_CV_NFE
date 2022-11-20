from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
import os
import random
import numpy as np
import pandas as pd


root = os.path.dirname(os.path.abspath(__file__))
properties = ["hair", "hair_length", "eyes"]
feature2svm = {}
class2value = {}
path = os.path.join(root, "attributes.csv")
attributes = pd.read_csv(path)
for i in properties:
    filename = f'svm_{i}.sav'
    svm = pickle.load(open(os.path.join(root, filename), 'rb'))
    feature2svm[i] = svm
    y = attributes[~attributes[i].isna()][i]
    label_encoder = LabelEncoder()
    label_encoder.fit(y)
    class2value[i] = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))


def get_boundary(vectors, attributes, feature):
    """
    Train the SVM for specified attribure
    :param vectors: np.array of latent vectors; shape=(N, z_dim)
    :param attributes: pd.DataFrame of all attributes; shape=(N, 5)
    :param feature: desired feature to train SVM on
    :return: trained SVM
    """
    X = vectors[~attributes[feature].isna()]
    y = attributes[~attributes[feature].isna()][feature]
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
    svm = SVC(kernel="linear", C=100)
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_train)
    print(f"Train accuracy: {accuracy_score(y_train, y_pred)}")
    y_pred = svm.predict(X_test)
    print(f"Test accuracy: {accuracy_score(y_test, y_pred)}")
    filename = f'svm_{feature}.sav'
    pickle.dump(svm, open(filename, 'wb'))
    return svm


def get_conditional_boundary(boundary, preserved_features=None):
    """
    make the hyperplane orthogonal to hyperplanes of specified features
    :param boundary: hyperplane of the feature to be changed
    :param preserved_features: list of feature names to be preserved
    :return: modified separating hyperplane
    """
    if preserved_features is None:
        return boundary / np.linalg.norm(boundary)
    cond = []
    for feature in preserved_features:
        cond.append(feature2svm[feature].coef_)
    cond = np.vstack(cond)
    A = np.matmul(cond, cond.T)
    B = np.matmul(cond, boundary.T)
    x = np.linalg.solve(A, B)
    new = boundary - (np.matmul(x.T, cond))
    return new / np.linalg.norm(new)


def manipulate(z, feature, value, preserved_features=None, start=-3.0, end=3.0, steps=6, z_dim=512):
    """
    Manipulate the image in the desired attribute
    :param z: latent code to be manipulated
    :param feature: feature to be changed
    :param value: desired target value of the feature
    :param preserved_features: features to be preserved
    :param start: The distance to the boundary where the manipulation starts
    :param end: The distance to the boundary where the manipulation ends
    :param steps: Number of manipulation steps between start and end
    :param z_dim: dimensionality of latent vector
    :return: list of manipulated latent vectors
    """
    if not os.path.exists(os.path.join(root, f'svm_{feature}.sav')):
        vectors = np.loadtxt(os.path.join(root, "vectors.csv"), delimiter=",")
        svm = get_boundary(vectors, attributes, feature)
    else:
        svm = feature2svm[feature]
    if len(z.shape) == 1:
        z = z.reshape((-1, z_dim))
    init_value = svm.predict(z)[0]
    value = class2value[feature][value]
    if value == init_value:
        value = random.choice([i for i in range(len(class2value[feature])) if i != init_value])
    boundary_idx = 0
    boundary_idx += sum([len(class2value[feature])-i-1 for i in range(min(init_value, value))])
    boundary_idx += abs(init_value-value)-1
    boundary = get_conditional_boundary(svm.coef_[boundary_idx], preserved_features)
    linspace = np.linspace(start, end, steps) - z.dot(boundary.T)
    return z + linspace.reshape(steps, 1) * boundary.reshape(1, -1)
