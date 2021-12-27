# -*- coding: utf-8 -*-
"""
Created on 2021.12.23

@author: zqr2008
"""

from time import time
from numpy.lib.function_base import disp
from sklearn import svm
##导入数据
import scipy.io as sio
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
import joblib
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
import xgboost
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import roc_auc_score
from scipy.stats import sem
import matplotlib.font_manager as fm
from sklearn import metrics
from prettytable import PrettyTable
import shap


myfont = fm.FontProperties(fname=r'C:\Windows\Fonts\AdobeSongStd-Light.otf') 


def eiras(filename1,filename2):
    bysy=pd.read_csv(filename1)
    szyy=pd.read_csv(filename2)
    train=bysy.values
    valid=szyy.values
    g=train.shape[1]

    X_train=train[:, :g-1]
    y_train=train[:,g-1]

    X_test=valid[0:, :g-1]
    y_test=valid[0:,g-1]  

    models = [
        {
    'label': 'Logistic Regression',
    'model': linear_model.LogisticRegression(penalty='l1',solver='liblinear'),
    },
        {
    'label':'XGboost',
    'model': XGBClassifier(subsample=1,eta=0.05,eval_metric=['logloss','auc','error'],use_label_encoder=False),
    },
        {
    'label': 'Gradient Boosting',
    'model': GradientBoostingClassifier(),    
    },
        {   'label':'Support Vector Machines',
    'model':svm.SVC(kernel='linear', gamma=1,C=1,probability=True),
    }           
        ]

    
    n_bootstraps = 1000
    rng_seed = 42  # control reproducibility
    bootstrapped_scores = []
    rng = np.random.RandomState(rng_seed)

    AUC=[]
    CI_lower=[]
    CI_upper=[]
    Specificity=[]
    Sensitivity=[]
    Accuracy=[]
    
    
    
    for m in models:
        model = m['model'] # select the model
        label=m['label']
        model.fit(X_train,y_train) # train the model
        y_pred=model.predict(X_test) # predict the test data
        y_pro=model.predict_proba(X_test)
        # Compute False postive rate=fpr, and True positive rate=tpr
        fpr, tpr, thresholds = metrics.roc_curve(y_test, model.predict_proba(X_test)[:,1])
        # compute sensitivity,specifictiy and accuracy
        tn, fp, fn, tp = confusion_matrix(y_test,y_pred,labels=model.classes_).ravel()
 
      
        sensitivity=float(tp) / float(tp+fn)
        specificity=float(tn) / float(tn+fp)
        accuracy=metrics.accuracy_score(y_test,y_pred, normalize=True, sample_weight=None)
        # Calculate Area under the curve to display on the plot
        auc = roc_auc_score(y_test,model.predict_proba(X_test)[:,1])
        plt.plot(fpr, tpr, label='%s ROC (area = %0.2f)' % (m['label'], auc))
        
        #confusion = confusion_matrix(y_test,y_pred)
        #disp = ConfusionMatrixDisplay(confusion_matrix=confusion,display_labels=model.classes_)
        #disp.plot()
        #plt.show()
   
        yy_pred=y_pro[:,1]
        for i in range(n_bootstraps):
            indices = rng.randint(0, len(yy_pred), len(yy_pred))  
            if len(np.unique(y_test[indices])) < 2:
                continue
            score = roc_auc_score(y_test[indices], yy_pred[indices])
            bootstrapped_scores.append(score)
        sorted_scores = np.array(bootstrapped_scores)
        sorted_scores.sort()
        confidence_lower = sorted_scores[int(0.05 * len(sorted_scores))]
        confidence_upper = sorted_scores[int(0.95 * len(sorted_scores))]
        print(label,"Confidence interval for the score: [{:0.3f} - {:0.3}]".format(confidence_lower, confidence_upper))
    
      
        Sensitivity.append(sensitivity)
        Specificity.append(specificity)
        Accuracy.append(accuracy)
        AUC.append(auc)
        CI_lower.append(confidence_lower)
        CI_upper.append(confidence_upper)

    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('1-特异率',fontproperties=myfont)
    plt.ylabel('敏感率',fontproperties=myfont)
    plt.legend(loc="lower right")
    plt.show()   
    return Sensitivity,Specificity,Accuracy,AUC,CI_upper,CI_lower




if __name__ == '__main__':
    filepath = ["C:/Users/mjdee/Desktop/JI-2020/ML/fortest0.csv","C:/Users/mjdee/Desktop/JI-2020/ML/result.csv"]
    Sensitivity,Specificity,Accuracy,AUC,CI_upper,CI_lower=eiras(filepath[0],filepath[1])
    the_table= PrettyTable(["Logistic Regression","XGboost","Gradient Boosting","Support Vector Machines"])
    the_table.add_row(Sensitivity)
    the_table.add_row(Specificity)
    the_table.add_row(Accuracy)
    the_table.add_row(AUC)
    the_table.add_row(CI_upper)
    the_table.add_row(CI_lower)
    print(the_table)
