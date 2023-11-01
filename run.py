import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("error") # if overflow during training / occurs

# base class for all layers of the neural network
class Layer:
    '''
    Base class for all layers of the neural network
    '''
    def __init__(self):
        self.X = None
        self.Y = None
    
    def forward_pass():
        pass

    def backward_pass():
        pass

class FullyConnected(Layer):
    def __init__(self, n_input, n_output):
        '''
        Initialize Fully Connected layer (a.k.a. FCLayer or Dense layer).

        Parameters:
        - n_input : int (a number of input neuron or features if it's the first layer).
        - n_output : int (a number of output neurons)
        '''
        self.W = np.random.rand(n_input, n_output) - 0.5 # randomly initializing weights
        self.B = np.random.rand(1, n_output) - 0.5 # randomly intiializing biases
    
    def forward_pass(self, X):
        '''
        Forward propagation step. Computes output Y and feeds forward it to the next layer.

        Parameters:
        - X : ndarray (input matrix)

        Returns:
        - Y : ndarray (output matrix)
        '''
        self.X = X # input
        self.Y = np.dot(self.X, self.W) + self.B # output calculated as X*W + B
        # print('forward', self.Y, '\n\n')
        return self.Y

    def backward_pass(self, dY, learning_rate, clip_norm):
        '''
        Backward propagation step. Calculates gradients and updates parameters.

        Parameters:
        - dY : ndarray (the gradient of the cost function with respect to the Y of the layer - dL/dY)
        - learning_rate : float (the rate at which the parameters should be updated each epoch)
        - clip_norm : float (constant threshold for gradient clipping)

        Returns:
        - dX : ndarray (i.e., dY for the previous layer)
        '''
        m = self.X.shape[0] # batch size for further normalizing of the parameters

        # gradient of the cost function with respect to the input X of the layer (dL/dX)
        # calculated as dY*W^T (considering chain rule)
        dX = np.dot(dY, self.W.T)

        # gradient of the cost function with respect to the weights of the layer (dL/dW)
        # calculated as X^T*dY divided by the number of samples (considering chain rule, 
        # and also the order matters following comutative property of matrix multiplication)
        dW = np.dot(self.X.T, dY) / m

        # gradient of the cost function with respect to the weights of the layer (dL/dB)
        # calculated as dY summed up divided by the number of samples (considering chain rule)
        dB = np.sum(dY, axis=0) / m

        dW_norm = np.linalg.norm(dW, ord=2) # l2-norm of vector dW
        dB_norm = np.linalg.norm(dB, ord=2) # l2-norm of vector dB
        # if the norm exceeds the threshold, clip(i.e. scale down) the gradient
        if dW_norm > clip_norm:
            dW = dW * clip_norm / dW_norm
        if dB_norm > clip_norm:
            dB = dB * clip_norm / dB_norm

        self.W = self.W - learning_rate*dW
        self.B = self.B - learning_rate*dB
        # print('backward', dW, '\n\n')
        return dX
    
class Activation(Layer):
    def __init__(self, Z, dZ):
        '''
        Initialize Activation layer.
        Takes an activation function and its derivative as the inputs.
        '''
        self.Z = Z
        self.dZ = dZ
    
    def forward_pass(self, X):
        '''
        Forward propagation step. Activates input and feeds forward it to the next layer.

        Parameters:
        - X : ndarray (input matrix)

        Returns:
        - Y : ndarray (output matrix)
        '''
        self.X = X
        self.Y = self.Z(self.X)
        return self.Y
    
    def backward_pass(self, dY, learning_rate=None, clip_norm=None):
        '''
        Backward propagation step.

        Parameters: 
        - dY : ndarray (the gradient of the cost function with respect to the Y of the layer - dL/dY)
        - learning_rate and clip_norm : float (left as a default argument due to convenience purposes in fit() function during training, has no impact here)
        
        Returns:
        - dZ(X)*dY : ndarray (the derivative of the activation function at points of input matrix X)
        '''
        return self.dZ(self.X) * dY
    
class ActivationFunction:
    '''
    A collection of static methods for different activation functions.
    Methods are static because they don't need to access or modify the class's state or instance data.
    These activation function classes under under ActivationFunction class just for organization and code modularity.
    '''
    class Sigmoid:
        @staticmethod
        def func(x):
            '''Compute the sigmoid of x.'''
            return 1 / (1 + np.exp(-x))
        
        @staticmethod
        def func_derivative(x):
            '''Compute the derivative of sigmoid, assuming x is sigmoid output.'''
            return x * (1 - x)
        
    class LeakyReLu:
        @staticmethod
        def func(x, alpha=0.01):
            '''Compute the Leaky ReLU of x.'''
            return np.where(x > 0, x, x * alpha)

        @staticmethod
        def func_derivative(x, alpha=0.01):
            '''Compute the derivative of Leaky ReLU, alpha is the slope for x < 0.'''
            return np.where(x > 0, 1, alpha)

    class HyperbolicTangent:
        @staticmethod
        def func(x):
            '''Compute the tanh of x.'''
            return np.tanh(x)
        
        @staticmethod
        def func_derivative(x):
            ''' Compute the derivative of tanh, assuming x is tanh output.'''
            return 1 - np.tanh(x)**2
        
class LossFunction:
    '''
    A collection of static methods for different loss functions.
    Methods are static because they don't need to access or modify the class's state or instance data.
    These loss function classes under under LossFunction class just for organization and code modularity.
    '''
    class BinaryLogLoss:
        @staticmethod
        def loss(Y, Y_hat, eps=1e-15):
            '''
            Compute binary cross-entropy.

            Parameters:
            - Y : ndarray (true labels)
            - Y_hat : ndarray (predicted labels)
            - eps : float (1e-15 by default, used for avoding log(0) and log(1))
            '''
            Y_hat = np.clip(Y_hat, eps, 1 - eps)
            m = Y.shape[0]
            L_sum = np.sum(np.multiply(Y, np.log(Y_hat)) + np.multiply(1 - Y, np.log(1 - Y_hat)))
            L = -(1/m) * L_sum
            return L
        
        @staticmethod
        def loss_derivative(Y, Y_hat, eps=1e-15):
            '''
            Compute the derivative binary cross-entropy.

            Parameters:
            - Y : ndarray (true labels)
            - Y_hat : ndarray (predicted labels)
            - eps : float (1e-15 by default, used for avoding log(0) and log(1))
            '''
            Y_hat = np.clip(Y_hat, eps, 1 - eps)
            L = -(Y/Y_hat) + (1 - Y)/(1 - Y_hat)
            return L
        
    class MeanSquaredError:
        @staticmethod
        def loss(Y, Y_hat):
            '''
            Compute mean squared error.

            Parameters:
            - Y : ndarray (true labels)
            - Y_hat : ndarray (predicted labels)
            '''
            return np.mean(np.power(Y - Y_hat, 2))
        
        @staticmethod
        def loss_derivative(Y, Y_hat):
            '''
            Compute the derivative of mean squared error.

            Parameters:
            - Y : ndarray (true labels)
            - Y_hat : ndarray (predicted labels)
            '''
            m = Y.shape[0]
            return 2*(Y_hat - Y)/m
        
class Model:
    '''
    A basic neural network model class with layers, loss function, 
    functionality for training and evaluating the model.
    '''
    def __init__(self):
        '''Initializes the model with no layers and no loss function'''
        self.layers = []
        self.loss = None
        self.loss_derivative = None
    
    def add(self, layer):
        '''
        Add a layer to the model.

        Parameters:
        - layer : object (a Layer object to be added to the model)
        '''
        self.layers.append(layer)
    
    def loss_type(self, loss, loss_derivative):
        '''
        Set the loss function and its derivative for the model.

        Parameters:
            loss : function (a LossFunction object's function to compute the loss)
            loss_derivative : function (a LossFunction object's function to compute the derivative of the loss)
        '''
        self.loss = loss
        self.loss_derivative = loss_derivative
    
    def fit(self, X_train, Y_train, n_epochs, learning_rate, clip_norm=1.0):
        '''
        Train the model on the provided dataset with specified hyperparameters.

        Parameters:
            X_train : ndarray (training features with (samples_number, features_number) shape)
            Y_train : ndarray (training labels with (samples_number, classes_number) for a classification task)
            n_epochs : int (number of epochs for training)
            learning_rate : float (learning rate for optimization)
        '''
        losses = []
        for epoch in range(n_epochs):
            Y_hat = X_train # the input for the first hidden layer
            for layer in self.layers:
                Y_hat = layer.forward_pass(Y_hat)
            # loss after forward passing all layers
            loss_value = self.loss(Y_train, Y_hat)
            losses.append(loss_value)
            
            # calculating dL/dY_hat derivative
            error = self.loss_derivative(Y_train, Y_hat)
            for layer in self.layers[::-1]:
                # print(f'epoch:{epoch}, layer:{layer}')
                error = layer.backward_pass(error, learning_rate, clip_norm)
        
            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Loss: {loss_value:.4f}')

    def predict(self, X_val):
        '''
        Predict the labels for the passed input data.

        Parameters:
        - X_val : ndarray (validation or test data to predict the output)
        
        Returns:
        - Y_val : ndarray (predicted output for the given input data)
        '''
        Y_val = X_val # the input for the first hidden layer
        for layer in self.layers:
            Y_val = layer.forward_pass(Y_val) # update output each forward pass
        return Y_val # final predicitons
    
    def calculate_confusion_matrix(self, Y, Y_hat):
        '''
        Calculate the confusion matrix for the given true labels and predictions.

        Parameters:
            Y : ndarray (true labels)
            Y_hat : ndarray (predicted labels)

        Returns:
            c_matrix : ndarray (the confusion matrix as a 2D numpy array)
        '''
        k = np.unique(Y).shape[0] # number of classes
        c_matrix = np.zeros((k,k))
        if k == 2:
            Y_hat = (Y_hat >= 0.5).astype(int)
        print('\n', pd.DataFrame(np.hstack([Y, Y_hat]), columns=['true', 'pred']))
        for i in range(Y.shape[0]):
            c_matrix[Y[i], Y_hat[i]] += 1
        return c_matrix

    def evaluate(self, y_val, predictions):
        '''
        Evaluate the model by calculating precision, recall, F1-score, and accuracy.

        Parameters:
            y_val : ndarray (true labels for the validation set)
            predictions : ndarray (predicted labels for the validation set)

        Returns:
            results : list (list containing the evaluation metrics results for each class)
        '''
        c_matrix = self.calculate_confusion_matrix(y_val, predictions)
        print('\n', pd.DataFrame(c_matrix, columns=[0, 1], index=[0, 1]))
        results = []
        accuracy = 0
        for i in range(c_matrix.shape[0]):
            tp = c_matrix[i, i]
            fp = c_matrix[:, i].sum() - tp
            fn = c_matrix[i, :].sum() - tp
            precision = tp / (tp + fp) if tp + fp != 0 else 0
            recall = tp / (tp + fn) if tp + fn != 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if precision + recall != 0 else 0
            results.append({
            'class': i,
            'precision': np.round(precision, 3),
            'recall': np.round(recall, 3),
            'f1': np.round(f1, 3)
            })
            accuracy += tp
            print(results[i])
        accuracy /= np.sum(c_matrix)
        print(f'accuracy: {accuracy}')
        return results
    
# training data
# x_train = np.array([
#                     [0, 0, 0],
#                     [0, 0, 1],
#                     [0, 1, 0],
#                     [0, 1, 1],
#                     [1, 0, 0],
#                     [1, 0, 1],
#                     [1, 1, 0],
#                     [1, 1, 1]])
# x_test = x_train
# y_train = np.array([
#                     [0],
#                     [0],
#                     [0],
#                     [0],
#                     [0],
#                     [0],
#                     [0],
#                     [1]])
# y_test = y_train

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