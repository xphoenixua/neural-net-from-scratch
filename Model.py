import numpy as np
import pandas as pd
import copy

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
        best_loss_ = np.inf
        self.losses_ = []
        for epoch in range(n_epochs):
            Y_pred = X_train # the input for the first hidden layer
            np.random.shuffle(Y_pred) # shuffle samples to prevent order learned biases
            for layer in self.layers:
                Y_pred = layer.forward_pass(Y_pred)

            # loss after forward passing all layers
            loss_value = self.loss(Y_train, Y_pred)
            self.losses_.append(loss_value)
            if loss_value < best_loss_:
                best_loss = loss_value
                best_model_ = copy.deepcopy(self.layers)

            # calculating dL/dY_pred derivative
            error = self.loss_derivative(Y_train, Y_pred)
            for layer in self.layers[::-1]:
                # print(f'epoch:{epoch}, layer:{layer}')
                error = layer.backward_pass(error, learning_rate, clip_norm)

            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Loss: {loss_value:.4f}')

        self.layers = best_model_

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
    
    def calculate_confusion_matrix(self, Y, Y_pred):
        '''
        Calculate the confusion matrix for the given true labels and predictions.

        Parameters:
            Y : ndarray (true labels)
            Y_pred : ndarray (predicted labels)

        Returns:
            c_matrix : ndarray (the confusion matrix as a 2D numpy array)
        '''
        k = np.unique(Y).shape[0] # number of classes
        c_matrix = np.zeros((k,k))
        print('\n', pd.DataFrame(np.hstack([Y, Y_pred]), columns=['true', 'pred']))
        for i in range(Y.shape[0]):
            c_matrix[Y[i], Y_pred[i]] += 1
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
        results = []
        accuracy = np.trace(c_matrix) / np.sum(c_matrix)
        for i in range(c_matrix.shape[0]):
            tp = c_matrix[i, i]
            fp = c_matrix[:, i].sum() - tp
            fn = c_matrix[i, :].sum() - tp
            precision = tp / (tp + fp) if tp + fp > 0 else 0
            recall = tp / (tp + fn) if tp + fn > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
            results.append({
                'class': i,
                'precision': precision,
                'recall': recall,
                'f1': f1
            })
        return results, accuracy