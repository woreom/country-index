import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix

from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression


import optuna

import joblib 
import matplotlib.pyplot as plt



def ensemble_classifier(Inputs, Targets, Forecast_Inputs, Labels, Index, hyper_tune=False, plotResults=True):
    if hyper_tune: 
        def CostFunc(trial):
        
            params = {  
                'svm_c': trial.suggest_float('svm_c', 0, 2),
                'rf_n_estimators':trial.suggest_int('rf_n_estimators', 5, 100),
                'rf_max_depth': trial.suggest_int('rf_max_depth', 2, 10),
                'mlp_hidden_layer': trial.suggest_categorical('mlp_hidden_layer', [(10,5), (10,), (5,), (20, ), (20, 10)]),
            }
            
            _,_, score=voting_classifier(Inputs, Forecast_Inputs, Labels, Index, params)
            
            return np.mean(score)
        
        study = optuna.create_study(direction='maximize', study_name='model Tuner' )
        study.optimize(CostFunc, n_trials=30)
        params=study.best_params 
    try: 
        params = joblib.load('ensemble_classifier_hyper_params.pkl')
    except:
        params = {  
            'svm_c': 0.8,
            'rf_n_estimators': 15,
            'rf_max_depth': 10,
            'mlp_hidden_layer': (10,)
        }

    outputs, forecast_outputs, _ =voting_classifier(Inputs, Forecast_Inputs, Labels, Index, params)
    y_forecast, y_pred, prob = logestic_classifier(outputs, Targets, forecast_outputs, plotResults)
    
    return y_forecast, y_pred, prob
    


def voting_classifier(Inputs, Forecast_Inputs, Labels, Index, params):


    classifiers = [
        ('svm', SVC(kernel='rbf',  probability=True, C=params['svm_c'])),
        ('rf', RandomForestClassifier(n_estimators=params['rf_n_estimators'], max_depth=params['rf_max_depth'])),
        ('mlp', MLPClassifier(hidden_layer_sizes=params['mlp_hidden_layer']))
    ]


    voting_clf = VotingClassifier(estimators=classifiers, voting='soft')
    
    outputs = []
    forecast_outputs = []
    score=[]


    for i, (X, X_forecast, y) in enumerate(zip(Inputs, Forecast_Inputs, Labels)):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        voting_clf.fit(X_train, y_train)

        accuracy = voting_clf.score(X_test, y_test)
        f1_weighted = f1_score(y_test, voting_clf.predict(X_test))
        score.append(f1_weighted)

        print(f'Group {i}: Accuracy: {accuracy}')
        

        pred = voting_clf.predict(X)

        outputs.append(pd.DataFrame(pred, index=Index[i]))

        forecast_vote = voting_clf.predict(X_forecast)
        forecast_outputs.append(forecast_vote)
    
    return outputs, forecast_outputs, score


def logestic_classifier(outputs, Targets, forecast_outputs, plotResults):


    # Combine the outputs from multiple models into a single dataframe
    X_ensemble = pd.concat([df for df in outputs], axis=1)

    # Convert the dataframe to a numpy array and back to a new dataframe
    X_ensemble = pd.DataFrame(X_ensemble.to_numpy(), index=X_ensemble.index)

    # Concatenate the target variable with the dataframe, drop any rows with missing values
    X_ensemble = pd.concat((X_ensemble, pd.DataFrame(Targets['label'])), axis=1).dropna()

    # Extract the input features and target variable from the dataframe
    X = X_ensemble[X_ensemble.columns[:-1]].to_numpy()
    y = X_ensemble[X_ensemble.columns[-1]].to_numpy()

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, shuffle=False)

    # Train a logistic regression classifier on the training data
    clf = LogisticRegression()
    clf.fit(X_train, y_train)

    # Predict the labels for the test data and calculate the accuracy of the classifier
    y_pred = clf.predict(X_test)
    accuracy = clf.score(X_test, y_test)
    print('Final Model Accuracy:', accuracy)
    
    if plotResults: plot_confusion_matrix(y_test, y_pred, list(np.unique(y_test)), normalize=True,title='Test')

    # Predict the label for the new set of input features
    x_forecast = [int(x[0]) for x in forecast_outputs]
    y_forecast = clf.predict(np.reshape(x_forecast,(1, -1)))
    
    y_pred = clf.predict(X)
    y_pred=pd.DataFrame({'Forecast': y_pred,'Real': y}, index=X_ensemble.index)
    
    prob=clf.predict_proba(np.reshape(x_forecast,(1, -1)))
    prob = np.max(prob[0])
    

    return y_forecast[0], y_pred, prob


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure()
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()
    
    


def plot_feature_importance(model, feature_names):
    """
    Given a trained LightGBM `model` and a list of `feature_names`, this function creates a Matplotlib bar chart
    showing the feature importance scores for the top 20 features.
    """
    importance_df = pd.DataFrame({'feature_name': feature_names, 'importance': model.feature_importances_})
    importance_df = importance_df.sort_values('importance', ascending=False).head(30)
    
    fig, ax = plt.subplots(figsize=(8, 10))
    ax.barh(importance_df['feature_name'], importance_df['importance'])
    ax.set_title("Feature Importance")
    ax.invert_yaxis()
    plt.show()

