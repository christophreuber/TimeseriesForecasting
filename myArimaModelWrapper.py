import itertools
import warnings
import math 

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import statsmodels.api as sm


def rmse(res):
    """Simple function to calculate the root mean squared eror."""
    mse = sum(res*res) / len(res)
    return math.sqrt(mse)


class ArimaModelWrapper:
    """Simplified access to arima models with somewhat automated training."""
    
    def __init__(self, data):
        self._data = data
        self._model = sm.tsa.statespace.SARIMAX(data)
        self._training_result = self._model.fit()
        
    def train(self, pmax=1, dmax=1, qmax=1, s=12, criterion='aic'):
        """Automatically trains multiple models over a grid of parameters.

        This is brute force, since we simply try a lot of parameter variations.
        The best model is selected based on the lowest value for the quality 
        criterion. 
        """
        
        # As a base line we use the current model.
        best_training_result = self._training_result
        try:
            best_fit = getattr(self._training_result, criterion)
        except AttributeError:
            print('Undefined criterion ' + criterion + '. Using RMSE instead.')
            best_fit = rmse(self._training_result.resid.values)
        
        # Generate all different combinations of p, d and q triplets with 
        # values for p,d,q between 0 and the given max value.
        p = range(0, pmax)
        d = range(0, dmax)
        q = range(0, qmax)
        pdq = list(itertools.product(p, d, q))
        seasonal_pdq = [(x[0], x[1], x[2], s) for x in list(itertools.product(p, d, q))]
        
        # We make a brute force approach here for simplicity and simply check 
        # all the parameter variations. Whenever a model has a better value
        # for the quality criterion we take it as the next best guess.
        warnings.filterwarnings("ignore") # specify to ignore warning messages
        for param in pdq:
            for param_seasonal in seasonal_pdq:
                try:
                    model = sm.tsa.statespace.SARIMAX(self._data,
                                                      order=param,
                                                      seasonal_order=param_seasonal,
                                                      enforce_stationarity=False,
                                                      enforce_invertibility=False)                
                except:
                    continue    # just try next parameter variation if this does not work
                training_result = model.fit()
                try:
                    current_fit = getattr(training_result, criterion)
                except AttributeError:
                    print('Undefined criterion ' + criterion + '. Using RMSE instead.')
                    current_fit = rmse(training_result.resid.values)
                if current_fit < best_fit:
                    best_fit = current_fit
                    best_training_result = training_result
                print('ARIMA{}x{}, {}:{}'.format(param, param_seasonal,
                                                 criterion, current_fit))
        self._training_result = best_training_result
        self._model = best_training_result.model
        print('Found best model:\nARIMA{}x{}, {}:{}'.format(self._model.order, 
                                                            self._model.seasonal_order, 
                                                            criterion, 
                                                            best_fit))
        
    def predict(self, start, end, dynamic=True):
        """Provides predictions for the trained model.
        
        start and end must be given as date string, e.g. '1970-06-01'.
        """
        
        if self._model is None:
            raise ValueError('I cannot predict anything if there is no model')            
        
        warnings.filterwarnings("ignore") # specify to ignore warning messages
        
        # if model is not trained yet or not up to date, then do so now
        if (self._training_result is None) \
            or (self._training_result.model != self._model):
            self._training_result = self._model.fit()
        
        pred = self._training_result.get_prediction(start=pd.to_datetime(start), 
                                                    end=pd.to_datetime(end), 
                                                    dynamic=dynamic, full_results=True)
        pred_mean = pred.predicted_mean
        pred_ci = pred.conf_int()
        return pred_mean, pred_ci
        
    def plot_prediction(self, start, end):
        """Plot predictions including confidence intervals"""
        
        pred_mean, pred_ci = self.predict(start, end)
        
        ax = pred_mean.plot(label='Forecast', alpha=0.7)
        ax.fill_between(pred_ci.index, pred_ci.iloc[:, 0], pred_ci.iloc[:, 1], 
                        label='Confidence interval', color='k', alpha=0.2)
        plt.legend()
        plt.show()
        
    
    def plot_diagnostics(self):
        if self._training_result is not None:
            self._training_result.plot_diagnostics(figsize=(15, 8))
            plt.show()
        
