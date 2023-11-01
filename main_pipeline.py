import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("error") # if overflow during training / occurs

# import custom neural network modules
from Model import *
from Layer import *
from ActivationFunction import *
from LossFunction import *

# Завантаження датасету Iris
iris = datasets.load_iris()
X_iris = iris.data[:100] # Тільки перші 100 записів (Setosa і Versicolour)
Y_iris = iris.target[:100].reshape(-1, 1) # Змінимо розмірність до (n_samples, 1)

# Стандартизація даних
scaler_iris = StandardScaler().fit(X_iris)
X_iris = scaler_iris.transform(X_iris)

# Розділення даних на навчальний та тестовий набори
x_train, x_test, y_train, y_test = train_test_split(X_iris, Y_iris,
test_size=0.2, random_state=42)

# design a neural network
m = x_train.shape[1] # number of features
n_hidden = [150]
n_output = y_train.shape[1] # number of classes

model = Model()
loss_fn = LossFunction.BinaryLogLoss
activation_fn_1 = ActivationFunction.HyperbolicTangent
activation_fn_2 = ActivationFunction.Sigmoid

model.loss_type(loss=loss_fn.loss, loss_derivative=loss_fn.loss_derivative)
model.add(FullyConnected(m, n_hidden[0]))
model.add(Activation(activation_fn_1.func, activation_fn_1.func_derivative))
model.add(FullyConnected(n_hidden[0], n_output))
model.add(Activation(activation_fn_2.func, activation_fn_2.func_derivative))

# train the net
model.fit(X_train=x_train, Y_train=y_train, 
          n_epochs=1000, 
          learning_rate=1e-5)

# get predictions
predictions = model.predict(x_test)

# validate it
results = model.evaluate(y_test, predictions)