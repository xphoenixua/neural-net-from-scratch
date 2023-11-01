import numpy as np

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