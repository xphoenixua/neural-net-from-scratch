import numpy as np

class ActivationFunction:
    '''
    A collection of static methods for different activation functions.
    Methods are static because they don't need to access or modify the class's state or instance data.
    These activation function classes are under ActivationFunction class just for organization and code modularity.
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
        
    class ReLU:
        @staticmethod
        def func(x):
            '''Compute the Leaky ReLU of x.'''
            return np.where(x >= 0, x, 0)

        @staticmethod
        def func_derivative(x):
            '''Compute the derivative of Leaky ReLU, alpha is the slope for x < 0.'''
            return np.where(x >= 0, 1, 0)

    class HyperbolicTangent:
        @staticmethod
        def func(x):
            '''Compute the tanh of x.'''
            return np.tanh(x)
        
        @staticmethod
        def func_derivative(x):
            ''' Compute the derivative of tanh, assuming x is tanh output.'''
            return 1 - np.tanh(x)**2
