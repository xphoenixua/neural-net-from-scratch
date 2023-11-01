import numpy as np

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