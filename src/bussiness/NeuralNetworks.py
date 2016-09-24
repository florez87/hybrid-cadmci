# -*- coding: utf-8 -*-

"""
Author: Juan Camilo Florez R. <florez87@gmail.com>
"""

from bussiness.Classifiers import Classifier
from keras.models import Sequential, load_model, save_model
from keras.layers import Dense
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix
from bussiness.Utilities import Utilities
from keras.utils import np_utils
import numpy

class NeuralNetwork(Classifier):
    """
    A neural network classifier based on keras Sequential.
    
    Parameters
    ----------
    database: string 
        The name of the database associated with the classifier.
        
    Attributes
    ----------
    trained: boolean
        Wether the model is already trained with any data.
        
    model: Sequential
        The neural network classifier from keras framework.
        
    classes: array of shape = [number_classes] 
        The list of arrays of class labels (multi-output problem).
    """
    def __init__(self, database):
        self.database = database
        self.trained = False
        self.model = Sequential()
        self.model.add(Dense(32, input_dim=4, init='uniform', activation='relu'))
        self.model.add(Dense(16, init='uniform', activation='relu'))
        self.model.add(Dense(3, init='uniform', activation='softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer='adamax', metrics=['accuracy'])
        self.classes = None
        
    def train(self, features, labels):
        """
        Train the model itself with a data set (features, labels).
        
        Parameters
        ----------
        features: array-like of shape = [number_samples, number_features]
            The training input samples.
            
        labels: array-like of shape = [number_samples, number_outputs]
            The target values (class labels in classification).
            
        Return
        ----------
        None
        """
        self.model.fit(features, labels, nb_epoch=250, batch_size=10, verbose=1)
        
    def predict(self, features):
        """
        Predict label and class probabilities of the input sample (features).
        
        Parameters
        ----------
        features: array-like of shape = [1, number_features]
            The input sample.
        
        Return
        ----------
        label: array of shape = [1]
            The predicted class for the input sample.
            
        probabilities: array of shape = [1, number_classes]
            The class probabilities for the input sample. The order of the
            classes corresponds to that in the attribute `classes`.
        """
        return self.classes[self.model.predict_classes(features, verbose = 0)], self.model.predict_proba(features, verbose = 0)
        
    def save(self, path, index):
        """
        Persist the model itself and it's classes with keras' save_model, joblib and pickle.
        
        Parameters
        ----------
        path: string
            The location of the persistence directory where model and classes will be stored.
        
        Return
        ----------
        None
        """
        save_model(self.model, path + index+ 'net.h5')
        joblib.dump(self.classes, path + index+ 'classes.pkl')
    
    def load(self, path, index):
        """
        Load a model and it's classes with keras' load_model, joblib and pickle.
        
        Parameters
        ----------
        path: string
            The location of the persistence directory from which model and classes will be loaded.
        
        Returns
        ----------
        None
        """
        self.model = load_model(path + index + 'net.h5')
        self.classes = joblib.load(path + index + 'classes.pkl')
        
    def validate(self, features, labels, number_folds, encoded_labels):
        """
        Compute a model's performance metrics based on k-fold cross-validation technique.
        
        Parameters
        ----------
        features: array-like of shape = [number_samples, number_features]
            The validation input samples.
            
        labels: array-like of shape = [number_samples]
            The target values (class labels in classification).
            
        number_folds: int
            The amount of folds for the k-fold cross-validation.
            If 0 compute metrics withput folds.
            If > 0 compute metrics with n folds, n=number_folds.
        
        encoded_labels: array-like of shape = [number_samples, number_outputs]
            The target values (class labels in classification) in one-hot-encoding.
            
        Return
        ----------
        accuracy: float
            The accuracy of the model based on it's confusion matrix.
            
        precision: float
            The precision of the model based on it's confusion matrix.
            
        sensitivity: float
            The sensitivity of the model based on it's confusion matrix.
            
        specificity: float
            The specificity of the model based on it's confusion matrix.
            
        kappa: float
            The Cohen's Kappa of the model based on it's confusion matrix.
        """
        if number_folds == 0:
            predictions = self.model.predict_classes(features)
        else:
            predictions = numpy.empty(len(labels), dtype=float)
            folds = Utilities.getFolds(labels, number_folds)
            for i, (train, test) in enumerate(folds):
                self.model.fit(features[train], encoded_labels[train], nb_epoch=250, batch_size=10, verbose=1)
                fold_prediction = self.model.predict_classes(features[test])
                for j in range(len(test)):
                    predictions[test[j]]=fold_prediction[j]
        matrix = confusion_matrix(np_utils.categorical_probas_to_classes(encoded_labels), predictions)
        sum_columns = numpy.sum(matrix, 0)
        sum_rows = numpy.sum(matrix, 1)
        diagonal_sum = numpy.trace(matrix)
        total_sum = numpy.sum(sum_rows)
        accuracy = diagonal_sum / total_sum
        temp_precision = []
        temp_sensitivity = []
        temp_specificity = []
        for i in range(len(matrix)):
            temp_precision.append(matrix[i][i] / sum_columns[i])
            temp_sensitivity.append(matrix[i][i] / sum_rows[i])
            temp_reduced_sum = total_sum - sum_rows[i] - sum_columns[i] + matrix[i][i]
            temp_specificity.append(temp_reduced_sum / (temp_reduced_sum + sum_columns[i] - matrix[i][i]))
        precision = sum(temp_precision * sum_rows) / total_sum
        sensitivity = sum(temp_sensitivity * sum_rows) / total_sum
        specificity = sum(temp_specificity * sum_rows) / total_sum
        kappa_sum = sum(sum_rows * sum_columns)
        kappa_numerator = (total_sum * diagonal_sum) - kappa_sum
        kappa_denominator =  (total_sum * total_sum) - kappa_sum
        kappa = kappa_numerator / kappa_denominator
        return accuracy, precision, sensitivity, specificity, kappa       