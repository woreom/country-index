from utils import (standardization, VMDdecomposition, FindBestLags,
                   find_heikin_ashi_candlestick, label_returns,subset_dataframes)


from investing import get_investing, update_investing, get_features
from metatrader import get_data_from_mt5, get_country_index_from_mt5, correct_candle


from ml_model import  ensemble_classifier
from deep_model import transformer_model

import numpy as np
import pandas as pd




def country_index_classifier(method, initialize, country='USD', TimeFrame='1d', 
                            use_haiken_ashi=True, hyper_tune=False, plot_results=False, force_to_train=False, update_fund_data=False):
    """
    Classifies the forecast and probability based on the given method and parameters.

    Parameters:
    - method (str): The classification method to use ('quant' or 'fund').
    - initialize: The initialization parameter for the classifier.
    - country (str): The country currency to consider (default: 'USD').
    - TimeFrame (str): The time frame to use for classification (default: '1d').
    - use_haiken_ashi (bool): Whether to use Haiken Ashi candles for classification (default: True).
    - hyper_tune (bool): Whether to perform hyperparameter tuning (default: False).
    - plot_results (bool): Whether to plot the classification results (default: False).
    - force_to_train (bool): Whether to force the classifier to train (default: False).
    - update_fund_data (bool): Wheter to Download and update fundamental data 
    Returns:
    - forecast (str): The forecast result ('sell', 'neutral', or 'buy').
    - prob (float): The probability of the forecast result.
    - outputs (dict): Outputs from the currency classifier.
    
    Raises:
    - ValueError: If an invalid method is specified.
    """

    if method == 'quant':
        if force_to_train:
            forecast, prob, net, acc, encoding, outputs = quant_classifier(initialize, currency=country, TimeFrame=TimeFrame,
                                                                use_pre_train=False, use_haiken_ashi=use_haiken_ashi,  
                                                                hyper_tune=hyper_tune, plot_results=plot_results)
        else:
            try:
                forecast, prob, net, acc, encoding, outputs = quant_classifier(initialize, currency=country, TimeFrame=TimeFrame,
                                                                    use_pre_train=True, use_haiken_ashi=use_haiken_ashi,  
                                                                    hyper_tune=hyper_tune, plot_results=plot_results)
            except:
                forecast, prob, net, acc, encoding, outputs = quant_classifier(initialize, currency=country, TimeFrame=TimeFrame,
                                                                    use_pre_train=False, use_haiken_ashi=use_haiken_ashi,  
                                                                    hyper_tune=hyper_tune, plot_results=plot_results)
        if forecast == 0:
            forecast = 'sell'
        elif forecast == 1:
            forecast = 'neutral'
        elif forecast == 2:
            forecast = 'buy'
        prob = round(prob, 2)

    elif method == 'fund':
        if TimeFrame not in ['1d', '1w']:
            raise ValueError('Invalid timeframe, should be daily(1d) or weekly(1w)')
        else:
            forecast, outputs, prob = fund_model(country, timeframe=TimeFrame,  
                                                use_haiken_ashi=use_haiken_ashi, 
                                                hyper_tune=hyper_tune, plot_results=plot_results,update_fund_data=update_fund_data)
            outputs = outputs.rename(columns={'Forecast': 'fund', 'Real': 'real'})
            outputs.replace(0, 'sell', inplace=True)
            outputs.replace(1, 'buy', inplace=True)

            if forecast == 0:
                forecast = 'sell'
            elif forecast == 1:
                forecast = 'buy'
      
    else:
        raise ValueError('Invalid method, should be quant or fund')

    return forecast, prob, outputs





def quant_classifier(initialize, currency='EURUSD', TimeFrame='15m',
            use_pre_train=True, use_haiken_ashi=True, hyper_tune=False, plot_results=False):
    

    Numimf, max_nsample, lagmax, nlags, n_trials=32, int(20e3), 200, 8, 50
    
    if len(currency)==3:
        df=get_country_index_from_mt5(initialize, currency, TimeFrame)
    else:
        
        try:
            df=correct_candle(initialize, currency, TimeFrame)
            
        except:
            df=get_data_from_mt5(initialize, currency, TimeFrame)
    
    if use_haiken_ashi:
        save_name=f"{df['info'].iloc[0]}_use_haiken_ashi"
    else:
        save_name=df['info'].iloc[0]

    n_sample = np.min([len(df), max_nsample]) 
    df=df.iloc[-n_sample:-1]
    
    if not (len(df) % 2 == 0) : df=df[1:]
    
    if use_haiken_ashi:
        y=label_returns(find_heikin_ashi_candlestick(df))
    else:
        y=label_returns(df)
    
    X=df['diff'].to_numpy()
    
    X=X.reshape(-1, 1)
    NumSamples = X.shape[0]
    NumFeature = X.shape[1]
    Inputs = np.zeros((NumSamples, NumFeature, Numimf))
    for i in range(NumFeature):
        Inputs[:, i, :] = VMDdecomposition(X[:, i], Numimf=Numimf, plotResults=plot_results)
    Inputs = Inputs.reshape(NumSamples, NumFeature * Numimf)

    X_scaled,_ = standardization(Inputs)
    
    inputs, inputs_forecast = FindBestLags(X_scaled,
                                                 y['diff'].to_numpy().reshape(-1,),
                                                 lagmax=lagmax,
                                                 nlags=nlags)
    labels = y['label'].iloc[-len(inputs):].to_numpy()

    forecast, prob, net, acc, encoding, label_indices=transformer_model(inputs, labels, inputs_forecast,
                                                               save_name=save_name, use_pre_train=use_pre_train, 
                                                               hyper_tune=hyper_tune, plotResults=plot_results, n_trials=n_trials)
    

    outputs=pd.DataFrame({'quant classifier': label_indices, 'label': labels }, index=y.index[-len(inputs):])
    outputs.replace(0, 'sell', inplace=True)
    outputs.replace(1, 'neutral', inplace=True)
    outputs.replace(2, 'buy', inplace=True)
    
    return  forecast, prob, net, acc, encoding, outputs





def fund_model(country , timeframe, features=None, method='clustering', subset_cols=None,
               use_haiken_ashi=True, hyper_tune=False, plot_results=False, update_fund_data=False):

    
    if features==None: features=get_features()[country]
    
    if update_fund_data: update_investing(method='update-country', country=country)
    X, y = get_investing(country=country, timeframe=timeframe)
    
    
    if method=='user_defined':
        
        # subset_cols=[
        # ['EURUSD', 'NZDUSD', 'GBPUSD', 'USDCHF'],
        # ['Silver', 'Gold', 'Copper', 'CRB'],
        # ['NASDAQ','VIX','T-Note','US 30 Cash', 'bond10y', 'bond2y', 'bond5y']
        # ]

        subsets=subset_dataframes(X, features, method='user_defined', subset_cols=subset_cols)
    else:
        subsets=subset_dataframes(X, features, method='clustering', n_clusters=3, subset_cols=None)

    Targets=y
    if use_haiken_ashi: Targets=find_heikin_ashi_candlestick(Targets)

    Targets['label'] = Targets['diff'].apply(lambda x: 1 if x > 0 else 0)

    Inputs = []
    Forecast_Inputs=[]
    Labels = []
    Index=[]
    for subset in subsets:
        dataset = pd.concat((subset, pd.DataFrame(Targets['diff']), pd.DataFrame(Targets['label'])), axis=1).dropna()
        X, y = dataset[dataset.columns[:-2]], dataset[dataset.columns[-2:]]
        
        X_scaled,_ = standardization(X.to_numpy())

        Best_Inputs, Forecast_Input=FindBestLags(X_scaled, y['diff'].to_numpy() , lagmax=200, nlags=10)
        Inputs.append(Best_Inputs.reshape((-1, Best_Inputs.shape[1]*Best_Inputs.shape[2])))
        Forecast_Inputs.append(Forecast_Input.reshape((-1, Forecast_Input.shape[1]*Forecast_Input.shape[2])))
        Labels.append(y['label'].to_numpy()[-len(Best_Inputs):])
        Index.append(dataset.index[-len(Best_Inputs):])
        
    forecast, y_pred, prob= ensemble_classifier(Inputs, Targets, Forecast_Inputs, Labels, Index, hyper_tune=hyper_tune, plotResults=plot_results)
    return forecast, y_pred, prob

def country_to_currency(y1, y2):
    if y1==y2: ye='neutral'
    else: ye=y1
    
    return ye



def fund_classifier(currency, timeframe='1w', use_haiken_ashi=True, hyper_tune=False, plot_results=False, update_fund_data=False):
    

    outputs=pd.DataFrame(columns=['forecast', 'prob'], index=[currency[:3], currency[3:], currency])
    
    preds=pd.DataFrame(columns=[currency[:3], currency[3:]])
    for country in [currency[:3], currency[3:]]:
        forecast, pred, prob=  fund_model(country, timeframe=timeframe, use_haiken_ashi=use_haiken_ashi,
                                          hyper_tune=hyper_tune, plot_results=plot_results, update_fund_data=update_fund_data)
        
        preds[country]=pred['Forecast']
        outputs['forecast'][country]=forecast
        outputs['prob'][country]=prob
    
    
    preds[currency] = preds.apply(lambda row: country_to_currency(row[currency[:3]], row[currency[3:]]), axis=1)
    preds.replace(0, 'sell', inplace=True)
    preds.replace(1, 'buy', inplace=True)
    

    if outputs['forecast'][0]==outputs['forecast'][1]:
        outputs['forecast'][currency]='neutral'
    else:
        outputs['forecast'][currency]=outputs['forecast'][0]
    
    outputs['prob'][currency]=outputs['prob'][0]*outputs['prob'][1]
    
    outputs.replace(0, 'sell', inplace=True)
    outputs.replace(1, 'buy', inplace=True)
    
    return outputs['forecast'][currency], outputs['prob'][currency], preds







    



    



