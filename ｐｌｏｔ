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
from sklearn.metrics import confusion_matrix
from sklearn.datasets import load_digits
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

myfont = fm.FontProperties(fname=r'C:\Windows\Fonts\AdobeSongStd-Light.otf') 




def eiras(filename1,filename2):
    bysy=pd.read_csv(filename1)
    szyy=pd.read_csv(filename2)
    train=bysy.values
    valid=szyy.values
    g=train.shape[1]
    number=valid.shape[0]

    X_train=train[:, :g-1]
    y_train=train[:,g-1]

    X_test=valid[0:, :g-1]
    y_test=valid[0:,g-1] 
    

    ##对数据进行特征缩放，对特征进行标准化处理
    sc = StandardScaler()     #实例化了一个StandardScaler对象，用sc引用
    sc.fit(X_train) 
     #使用StandardScaler中的fit方法，可以计算训练数据中每个特征的μ（样本均值）和σ（标准差）
    #通过调用transform方法，可以使用前面计算得到的μ和σ来对训练数据做标准化处理。
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)

    
    LOGI = linear_model.LogisticRegression(penalty='l1',solver='liblinear')
    XGBO = XGBClassifier(subsample=1,eta=0.05,eval_metric=['logloss','auc','error'])
    GBDT = GradientBoostingClassifier()
    SVM = svm.SVC(kernel='linear', gamma=1,C=1,probability=True) 
    
    t0 = time()
    GBDT.fit(X_train,y_train)
    t1 = time()

    
    
    ##预测：计算分类错误的个数
    y_pred = GBDT.predict(X_test)
    y_pro=GBDT.predict_proba((X_test))
    
    n_bootstraps = 1000
    rng_seed = 42  # control reproducibility
    bootstrapped_scores = []
    rng = np.random.RandomState(rng_seed)
    
   
    yy_pred=y_pro[:,1]

    for i in range(n_bootstraps):
        # bootstrap by sampling with replacement on the prediction indices
         indices = rng.randint(0, len(yy_pred), len(yy_pred))
         if len(np.unique(y_test[indices])) < 2:
            continue
         score = roc_auc_score(y_test[indices], yy_pred[indices])
         bootstrapped_scores.append(score)
   
    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()
    confidence_lower = sorted_scores[int(0.05 * len(sorted_scores))]
    confidence_upper = sorted_scores[int(0.95 * len(sorted_scores))]
    print("Confidence interval for the score: [{:0.3f} - {:0.3}]".format(confidence_lower, confidence_upper))


    i=-1
    j=-1
    y_pred=np.zeros(325) #1个数组，313个0
    yf_GBDT=np.zeros([3,500]) #3个数组，每个数组500个0
    yi=np.linspace(0,1,500) #500个数，为0到1的等差序列
    while (j<499):
      j=j+1
      i=-1
      while (i<324):
          i=i+1
          if y_pro[i,0]<yi[j]:
              y_pred[i]=1;
          if y_pro[i,0]>yi[j]:
              y_pred[i]=0;
            
    
      #print('Accuracy: %.4f' % accuracy_score(y_test,y_pred))
      #print('Accuracy1: %.4f' % accuracy_score(y_test,y_pred1))
      yf_GBDT[0,j]=accuracy_score(y_test,y_pred)
      tn, fp, fn, tp = confusion_matrix(y_test,y_pred).ravel()
      Sensitivity = float(tp)/float(tp+fn) 
      specificity = float(tn) / float(tn+fp)
    
      tn1, fp1, fn1, tp1 = confusion_matrix(y_test,y_pred).ravel()
      Sensitivity1 = float(tp1)/float(tp1+fn1) 
      specificity1 = float(tn1) / float(tn1+fp1)
      yf_GBDT[1,j]=Sensitivity1
      yf_GBDT[2,j]=specificity1;

    #logistic regression
    t0 = time()
    LOGI.fit(X_train,y_train)
    t1 = time()
    

    y_predLOGI = GBDT.predict(X_test)
    y_proLOGT=GBDT.predict_proba((X_test))

    i=-1
    j=-1
    y_predLOGI=np.zeros(325) #1个数组，313个0
    yf_LOGI=np.zeros([3,500]) #3个数组，每个数组500个0
    yiLOGI=np.linspace(0,1,500) #500个数，为0到1的等差序列
    while (j<499):
      j=j+1
      i=-1
      while (i<324):
          i=i+1
          if y_proLOGT[i,0]<yi[j]:
              y_predLOGI[i]=1;
          if y_proLOGT[i,0]>yi[j]:
              y_predLOGI[i]=0;
            
    
  
      yf_LOGI[0,j]=accuracy_score(y_test,y_predLOGI)
      tn, fp, fn, tp = confusion_matrix(y_test,y_predLOGI).ravel()
      Sensitivity = float(tp)/float(tp+fn) 
      specificity = float(tn) / float(tn+fp)
    
      tn1, fp1, fn1, tp1 = confusion_matrix(y_test,y_predLOGI).ravel()
      Sensitivity1 = float(tp1)/float(tp1+fn1) 
      specificity1 = float(tn1) / float(tn1+fp1)
      yf_LOGI[1,j]=Sensitivity1
      yf_LOGI[2,j]=specificity1;



#XGBO
    t0 = time()
    XGBO.fit(X_train,y_train)
    t1 = time()
    

    y_predXGBO= XGBO.predict(X_test)
    y_proXGBO=XGBO.predict_proba((X_test))

    i=-1
    j=-1
    y_predXGBO=np.zeros(325) #1个数组，313个0
    yf_XGBO=np.zeros([3,500]) #3个数组，每个数组500个0
    yiXGBO=np.linspace(0,1,500) #500个数，为0到1的等差序列
    while (j<499):
      j=j+1
      i=-1
      while (i<324):
          i=i+1
          if y_proXGBO[i,0]<yi[j]:
              y_predXGBO[i]=1;
          if y_proXGBO[i,0]>yi[j]:
              y_predXGBO[i]=0;
            
    
  
      yf_XGBO[0,j]=accuracy_score(y_test,y_predXGBO)
      tn, fp, fn, tp = confusion_matrix(y_test,y_predXGBO).ravel()
      Sensitivity = float(tp)/float(tp+fn) 
      specificity = float(tn) / float(tn+fp)
    
      tn1, fp1, fn1, tp1 = confusion_matrix(y_test,y_predXGBO).ravel()
      Sensitivity1 = float(tp1)/float(tp1+fn1) 
      specificity1 = float(tn1) / float(tn1+fp1)
      yf_XGBO[1,j]=Sensitivity1
      yf_XGBO[2,j]=specificity1;  
    

#SVM 
    t0 = time()
    SVM.fit(X_train,y_train)
    t1 = time()
    

    y_predSVM= SVM.predict(X_test)
    y_proSVM=SVM.predict_proba((X_test))

    i=-1
    j=-1
    y_predSVM=np.zeros(325) #1个数组，313个0
    yf_SVM=np.zeros([3,500]) #3个数组，每个数组500个0
    yiSVM=np.linspace(0,1,500) #500个数，为0到1的等差序列
    while (j<499):
      j=j+1
      i=-1
      while (i<324):
          i=i+1
          if y_proSVM[i,0]<yi[j]:
              y_predSVM[i]=1;
          if y_proSVM[i,0]>yi[j]:
              y_predSVM[i]=0;
            
    
  
      yf_SVM[0,j]=accuracy_score(y_test,y_predSVM)
      tn, fp, fn, tp = confusion_matrix(y_test,y_predSVM).ravel()
      Sensitivity = float(tp)/float(tp+fn) 
      specificity = float(tn) / float(tn+fp)
    
      tn1, fp1, fn1, tp1 = confusion_matrix(y_test,y_predSVM).ravel()
      Sensitivity1 = float(tp1)/float(tp1+fn1) 
      specificity1 = float(tn1) / float(tn1+fp1)
      yf_SVM[1,j]=Sensitivity1
      yf_SVM[2,j]=specificity1;

    plt.plot([0, 1], [0, 1], 'k--')
    disp=plot_roc_curve(LOGI,X_test,y_test)
    plot_roc_curve(GBDT,X_test,y_test,ax=disp.ax_)
    plot_roc_curve(XGBO,X_test,y_test,ax=disp.ax_)
    plot_roc_curve(SVM,X_test,y_test,ax=disp.ax_)
    plt.legend(loc='best')
    plt.ylabel('敏感率',fontproperties=myfont)
    plt.xlabel('1-特异率',fontproperties=myfont)
    plt.show()
       

    sen=yf_LOGI[1,:]
    spe=1-yf_LOGI[2,:]
    AUC_LOGI = np.trapz(sen,spe)
    mean_LOGI=np.mean(yf_LOGI[0,:])

    sen=yf_XGBO[1,:]
    spe=1-yf_XGBO[2,:]
    AUC_XGBO = np.trapz(sen,spe)
    mean_XGBO=np.mean(yf_XGBO[0,:])

    sen=yf_SVM[1,:]
    spe=1-yf_SVM[2,:]
    AUC_SVM = np.trapz(sen,spe)
    mean_SVM=np.mean(yf_SVM[0,:])

    sen=yf_GBDT[1,:]
    spe=1-yf_GBDT[2,:]
    AUC_GBDT = np.trapz(sen,spe)
    mean_GBDT=np.mean(yf_GBDT[0,:])
    

    #print('accuracy:',mean_GBDT)
    print('AUC of GBDT is',AUC_GBDT,'AUC of Logistic Regression is',AUC_LOGI
    ,'AUC of XGBOOST is',AUC_XGBO,'AUC of SVM is',AUC_SVM )
    
    score, ci_lower, ci_upper, scores = stat_util.score_ci(
   y_true, y_pred, score_fun = roc_auc_score)

    #旧作图方法   
    #plt.figure(1).clf() 
    #plt.rc('font',family='Times New Roman')
    #plt.plot([0, 1], [0, 1], 'k--')
    #plt.ylabel('Sensitivity rate')
    #plt.xlabel('1-specificity rate')
    #plt.plot(1-yf_GBDT[2,:], yf_GBDT[1,:],label='GBDT')
    #plt.plot(1-yf_LOGI[2,:], yf_LOGI[1,:], label='LOGI')
    #plt.legend(loc='best')
 
    #print(GBDT.feature_importances_)
    #plt.figure(2)
    #plt.plot(75-fir,the_curve[75-fir]) 
    return sen,spe,mean_GBDT,AUC_GBDT


if __name__ == '__main__':
    filepath = ["C:/Users/mjdee/Desktop/JI-2020/ML/test(1).csv","C:/Users/mjdee/Desktop/JI-2020/ML/result.csv"]
    sen,spe,mean_GBDT,AUC_GBDT=eiras(filepath[0],filepath[1])
    #szyy=pd.read_csv(filepath[1])
    #valid=szyy.values
    #X_test=valid[0:,:]
    #print(X_test.shape)
