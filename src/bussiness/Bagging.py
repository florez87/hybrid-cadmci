# -*- coding: utf-8 -*-

"""
Author: Juan Camilo Florez R. <florez87@gmail.com>
"""

from bussiness.Classifiers import Classifier
from bussiness.DecisionTrees import DecisionTree
from bussiness.NeuralNetworks import NeuralNetwork
from bussiness.Utilities import Utilities
from sklearn.utils import check_random_state
from keras.utils import np_utils
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix
import numpy

class Bagging(Classifier):
    """
    A bagging classifier based on 5 decision trees and 5 neural networks.
    
    Parameters
    ----------
    database: string 
        The name of the database associated with the classifier.
        
    Attributes
    ----------
    trained: boolean
        Wether the model is already trained with any data.
        
    model: array of shape = [number_classifiers]
        Empty array that will hold the models comprising the bagging classifier.
        
    number_classifiers: int, default(10)
        The amount of classifiers for bagging.
        
    classes: array of shape = [number_classes] 
        The list of arrays of class labels (multi-output problem).
        
    References
    ----------
    .. [1] Hsu, K.-W. (2012). Hybrid Ensembles of Decision Trees and Artificial 
        Neural Networks. CyberneticsCom, 25-29.
    """
    def __init__(self, database):
        self.database = database
        self.trained = False
        self.model = []
        self.number_classifiers = 11
        self.classes = None
        for i in range(self.number_classifiers):
            if i % 2 != 0:
                model = DecisionTree(database)
            else:
                model = NeuralNetwork(database)
            self.model.append(model)
    
    def train(self, features, labels):
        """
        Train the model itself with a data set (features, labels).
        The training method uses bootstrapping to extract samples from the data set.
        
        Parameters
        ----------
        features: array-like of shape = [number_samples, number_features]
            The training input samples.
            
        labels: array-like of shape = [number_samples]
            The target values (class labels in classification).
            
        Return
        ----------
        None
        """
        number_samples, number_features = features.shape
        max_samples = number_samples
        random_state = check_random_state(None)
        seeds = random_state.randint(numpy.iinfo(numpy.int32).max, size=self.number_classifiers)
        classes, ids = Utilities.getClasses(labels)
        encoded_labels = np_utils.to_categorical(ids, len(classes))
        for i in range(self.number_classifiers):
            random_state = check_random_state(seeds[i])
            indices = random_state.randint(0, number_samples, max_samples)
            if i % 2 != 0:
                self.model[i].train(features[indices], labels[indices])
                self.model[i].classes = self.model[i].model.classes_
            else:
                self.model[i].train(features[indices], encoded_labels[indices])
                self.model[i].classes = classes
        self.classes = classes
        
    def predict(self, features):
        """
        Predict label and class probabilities of the input sample (features).
        Predictions are obtained by voting or averaging each model estimate.
        
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
        predictions = []
        predictions_votes = numpy.empty(len(self.classes), dtype=object)
        predictions_probas = []
        final_proba = []
        max_class = None
        for i in range(self.number_classifiers):
            temp_prediction, temp_proba = self.model[i].predict(features)
            predictions.append(temp_prediction)
            predictions_probas.append(temp_proba)
        predictions_votes[self.classes.tolist().index("Sane")] = predictions.count('Sane')
        predictions_votes[self.classes.tolist().index("Mild")] = predictions.count('Mild')
        predictions_votes[self.classes.tolist().index("Serious")] = predictions.count('Serious')
        max_votes = max(predictions_votes)
        voting = [i for i, j in enumerate(predictions_votes) if j == max_votes]
        if(len(voting) > 1):
            final_proba = numpy.mean(predictions_probas, axis = 0)
            max_class_index = numpy.argmax(final_proba)
            max_class = [self.classes[max_class_index]]
        else: 
            max_class = self.classes[voting]
            temp_proba = []
            for i in range(len(predictions_votes)):
               temp_proba.append(predictions_votes[i]/self.number_classifiers)
            final_proba.append(temp_proba)
            final_proba = numpy.array(final_proba)
        return max_class, final_proba
        
    def save(self, path, index = 0):
        """
        Persist the 10 bagging models and it's classes.
        
        Parameters
        ----------
        path: string
            The location of the persistence directory where models and classes will be stored.
        
        Return
        ----------
        None
        """
        for i in range(self.number_classifiers):
            self.model[i].save(path, str(i))
        joblib.dump(self.classes, path + 'bagging_classes.pkl')
    
    def load(self, path, index = 0):
        """
        Load the 10 bagging models and it's classes.
        
        Parameters
        ----------
        path: string
            The location of the persistence directory from which models and classes will be loaded.
        
        Returns
        ----------
        None
        """
        for i in range(self.number_classifiers):
            self.model[i].load(path, str(i))
        self.classes = joblib.load(path + 'bagging_classes.pkl')
            
    def validate(self, features, labels, number_folds):
        """
        Compute bagging model's performance metrics based on k-fold cross-validation technique.
        
        Parameters
        ----------
        features: array-like of shape = [number_samples, number_features]
            The validation input samples.
            
        labels: array-like of shape = [number_samples] or [number_samples, number_outputs]
            The target values (class labels in classification).
            
        number_folds: int
            The amount of folds for the k-fold cross-validation.
            If 0 compute metrics withput folds.
            If > 0 compute metrics with n folds, n=number_folds.
        
        Return
        ----------
        accuracy: float
            The accuracy of the bagging model based on it's confusion matrix.
            
        precision: float
            The precision of the bagging model based on it's confusion matrix.
            
        sensitivity: float
            The sensitivity of the bagging model based on it's confusion matrix.
            
        specificity: float
            The specificity of the bagging model based on it's confusion matrix.
            
        kappa: float
            The Cohen's Kappa of the bagging model based on it's confusion matrix.
        """
        number_samples, number_features = features.shape
        predictions = []
        if number_folds == 0:
            for i in range(number_samples):
                prediction, _ = self.predict(features[i].reshape(1, -1))
                predictions.append(prediction)
        else:
            predictions = numpy.empty(len(labels), dtype=object)
            folds = Utilities.getFolds(labels, number_folds)
            for i, (train, test) in enumerate(folds):
                self.train(features[train], labels[train])                    
                for j in range(len(test)):
                    fold_prediction, _ = self.predict(features[test[j]].reshape(1, -1))
                    predictions[test[j]]=fold_prediction[0]
        matrix = confusion_matrix(labels, predictions)
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
##        
