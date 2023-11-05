import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import warnings
# warnings.filterwarnings("error") # if overflow during training / occurs

# import custom neural network modules
from Model import *
from Layer import *
from ActivationFunction import *
from LossFunction import *

# loading the preprocessed Titanic dataset from feature_engineering.ipynb (https://www.kaggle.com/competitions/titanic)
X_train = pd.read_csv('data/X_train.csv', index_col=0).values; Y_train = pd.read_csv('data/Y_train.csv', index_col=0).values
X_test = pd.read_csv('data/X_test.csv', index_col=0).values; Y_test = pd.read_csv('data/Y_test.csv', index_col=0).values

# design a neural network
m = X_train.shape[1] # number of features
n_hidden = [100, 200]
n_output = Y_train.shape[1] # number of classes

np.random.seed(42)
model = Model()
loss_fn = LossFunction.BinaryLogLoss
activation_fn_1 = ActivationFunction.HyperbolicTangent
activation_fn_2 = ActivationFunction.Sigmoid

model.loss_type(loss=loss_fn.loss, loss_derivative=loss_fn.loss_derivative)
model.add(FullyConnected(m, n_hidden[0]))
model.add(Activation(activation_fn_1.func, activation_fn_1.func_derivative))
model.add(FullyConnected(n_hidden[0], n_hidden[1]))
model.add(Activation(activation_fn_1.func, activation_fn_1.func_derivative))
model.add(FullyConnected(n_hidden[1], n_output))
model.add(Activation(activation_fn_2.func, activation_fn_2.func_derivative))

# train the net
model.fit(X_train=X_train, Y_train=Y_train, 
          n_epochs=500, 
          learning_rate=1e-5)
history = model.losses_
# get predictions
Y_pred = model.predict(X_test)
Y_pred = (Y_pred >= 0.5).astype(int)
mispreds = np.where(Y_pred != Y_test)[0]
# validate it
results, accuracy = model.evaluate(Y_test, Y_pred)
for cl in results:
    print(cl)
print('accuracy', accuracy)

test_df = pd.read_csv('data/test.csv')
test_df = test_df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1).dropna()
train_df = pd.read_csv('data/train.csv')
train_df = train_df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1).dropna()
print('\nMisprediction indices:', mispreds)
print(test_df.iloc[mispreds, :])
print(train_df.mean(), '\n', train_df['Embarked'].value_counts())

# # sklearn implementation
# clf = MLPClassifier(hidden_layer_sizes=n_hidden, 
#                     max_iter=500,
#                     solver='sgd',
#                     learning_rate_init=1e-5,
#                     activation='tanh',
#                     verbose=0, 
#                     random_state=42)
# clf.fit(X_train, Y_train.ravel())
# history = clf.loss_curve_
# Y_pred_sklearn = clf.predict(X_test)
# print('\n', classification_report(Y_test.ravel(), Y_pred_sklearn, zero_division=0))



# plot loss curve
fig, ax = plt.subplots()

ax.set_xlabel('epochs')
ax.set_ylabel('loss')
ax.plot(history)

# # regression line for the loss curve
# x = np.arange(len(history))
# k, b = np.polyfit(x, history, 1)
# reg_curve = k*x + b
# ax.plot(reg_curve)

plt.show()
