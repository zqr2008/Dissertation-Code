# -*- coding: utf-8 -*-
"""
Created on 2021.12.23

@author: zqr2008
"""

from time import time
from numpy.lib.function_base import disp
from sklearn import svm
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
from sklearn import preprocessing
from sklearn import model_selection
from sklearn.calibration import calibration_curve
from sklearn.calibration import CalibrationDisplay
import shap
from matplotlib.gridspec import GridSpec
from sklearn.calibration import CalibratedClassifierCV

myfont = fm.FontProperties(fname=r'C:\Windows\Fonts\simsun.ttc') 
myfont_times = fm.FontProperties(fname=r'C:\Windows\Fonts\times.ttf') 

def eiras(filename1,filename2):
    """
    Getting result of applying one file model on another.
    Arguments:two different datasets,filename1 for training set, filename2 for test set.
    Output: parameters that measure the performance of the exsiting model using filename1.
    """
    
    bysy=pd.read_csv(filename1)
    szyy=pd.read_csv(filename2,encoding='GBK')
    
    test=szyy
    szyy=test.iloc[:,2:]
    patient=test.iloc[:,0:2]
    
    train=bysy.values
    valid=szyy.values
    g=train.shape[1]
    
    

    X_train=train[:, :g-1]
    y_train=train[:,g-1]

    X_test=valid[:, :g-1]
    y_test=valid[:,g-1]  
    


    models = [
        {
    'label': 'LR',
    'model': linear_model.LogisticRegression(penalty='l1',solver='liblinear'),
    'rank': 'B',
    },
        {
     'label':'SVM',
    'model':svm.SVC(kernel='linear', gamma=1,C=1,probability=True),
    'rank': 'C',
    },
        {
    'label': 'GBDT',
    'model': GradientBoostingClassifier(),
    'rank': 'D',    
    },
        {
    'label':'XGboost',
    'model': XGBClassifier(subsample=1,eta=0.05,eval_metric=['logloss','auc','error'],use_label_encoder=False),
    'rank': 'E',
    }     
        ]
    
    
    def Find_Optimal_Cutoff(TPR, FPR, threshold):
        y = TPR - FPR
        Youden_index = np.argmax(y)  # Only the first occurrence is returned.
        optimal_threshold = threshold[Youden_index]
        point = [FPR[Youden_index], TPR[Youden_index]]
        return optimal_threshold,point
    
    def importance(model,X_train,feature_name):
        try:
            shap_values = shap.TreeExplainer(model).shap_values(X_train)
            #shap.summary_plot(shap_values, X_train,feature_names=feature_name)
            #shap.plots.bar(shap_values)
        except Exception:
            return None


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
    Optimal=[]
    Point=[]
    Youden=[]
    Predict_prob={}
    Predict_result={}



    
    for m in models:
        model = m['model'] # select the model
        label=m['label']
        model.fit(X_train,y_train) # train the model
        y_pred=model.predict(X_test) # predict the test data
        y_pro=model.predict_proba(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        # Compute False postive rate=fpr, and True positive rate=tpr
        fpr, tpr, thresholds = metrics.roc_curve(y_test, model.predict_proba(X_test)[:,1])
        # Calculate Area under the curve to display on the plot
        auc = roc_auc_score(y_test,model.predict_proba(X_test)[:,1])
        plt.plot(fpr, tpr, label='%s (AUC = %0.2f)' % (m['label'], auc))
        
        #using optimal_threshold function to find best sensitivity and specificity trade-off
        optimal_threshold,point=Find_Optimal_Cutoff(tpr,fpr,thresholds)
        best_y_pred = (model.predict_proba(X_test)[:,1] >= optimal_threshold).astype(bool)
        #fop, mpv = calibration_curve(y_test, y_prob, n_bins=5,strategy="quantile")
        #plt.plot([0, 1], [0, 1], linestyle='--')
        #plt.plot(mpv, fop, marker='.')
        #plt.show()
        
        Predict_prob[label]=y_prob
        
        calibrated_clf = CalibratedClassifierCV(base_estimator=model,cv="prefit")
        calibrated_clf.fit(X_test, y_test)
        calibrated_pred=calibrated_clf.predict(X_test)
        Predict_prob[label+label]=calibrated_pred
        Predict_prob[label+label+label]=best_y_pred
        # compute sensitivity,specifictiy and accuracy
        #confusion = confusion_matrix(y_test,best_y_pred)
        #disp = ConfusionMatrixDisplay(confusion_matrix=confusion,display_labels=model.classes_)
        #disp.plot()
        #plt.show()
        
        tn, fp, fn, tp = confusion_matrix(y_test,best_y_pred).ravel()
        sensitivity=float(tp) / float(tp+fn)
        specificity=float(tn) / float(tn+fp)
        youden=(sensitivity+specificity)-1
        accuracy=metrics.accuracy_score(y_test,best_y_pred, normalize=True, sample_weight=None)
        
        #importance
       
        #importance(model,X_train,feature_name
        
        
      
        #compute CI using bootstraps
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
        Youden.append(youden)
        Optimal.append(optimal_threshold)
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('1-特异率',fontproperties=myfont)
    plt.ylabel('敏感率',fontproperties=myfont)
    plt.legend(loc="lower right")
    #plt.show() 
    plt.savefig(r"C:\Users\mjdee\Desktop\JI-2020\ML\the_roc_simple.jpg", bbox_inches='tight',dpi=600)
    plt.close()  
    
    #plot calibration_curve
    fig = plt.figure(figsize=(10, 10))
    
    gs = GridSpec(4, 2)
    ax_calibration_curve = fig.add_subplot(gs[:2, :2])
    calibration_displays = {}
    colors = plt.cm.get_cmap("Dark2")
    i=0
    
    for m in (models):
        model = m['model']
        label= m['label']
        model.fit(X_train, y_train)
        display = CalibrationDisplay.from_estimator(
        model,
        X_test,
        y_test,
        n_bins=5,
        strategy="quantile",
        name=label,
        ax=ax_calibration_curve,
        color=colors(i),
    )
        i=i+1
        calibration_displays[label] = display
        
    ax_calibration_curve.set_xlim(0,0.5)
    ax_calibration_curve.grid()
    ax_calibration_curve.set_title("A       Calibration plots",fontweight="bold", size=18)
    
    #plot calibration_histograms    
    grid_positions = [(2, 0), (2, 1), (3, 0), (3, 1)]
    i=0
    for m in (models):
        model = m['model']
        label= m['label']
        rank=m['rank']
        row, col = grid_positions[i]
        ax = fig.add_subplot(gs[row, col])

        ax.hist(
            calibration_displays[label].y_prob,
            range=(0, 1),
            bins=5,
            label=name,
            color=colors(i)
            )
        i=i+1
        ax.set_title(str(rank)+"      "+str(label),fontweight="bold", size=18)
        ax.set_ylabel('Count', fontsize = 18.0)
        ax.set_xlabel('Mean predicted probability', fontsize = 18)
        #ax.set(title=str(rank)+"      "+str(label),xlabel="Mean predicted probability", ylabel="Count")

    plt.tight_layout()
    plt.savefig(r"C:\Users\mjdee\Desktop\JI-2020\ML\simple_calibration.jpg", bbox_inches='tight',dpi=600)
    
    
    return Sensitivity,Specificity,Accuracy,AUC,CI_upper,CI_lower,Youden,Optimal,Predict_result,Predict_prob,X_test,y_test,patient


if __name__ == '__main__':
    filepath = [r"C:\Users\mjdee\Desktop\JI-2020\ML\test(convert_alive_7d).csv",r"C:\Users\mjdee\Desktop\JI-2020\ML\PUTH2.csv"]
    name=(["逻辑回归","支持向量机","梯度提升树","极端梯度提升树"])
    
    Sensitivity,Specificity,Accuracy,AUC,CI_upper,CI_lower,Youden,Optimal,predict_result,predict_prob,X_test,y_test,patient=eiras(filepath[0],filepath[1])
    
    dataoutput={'机器学习算法':name,'敏感率': Sensitivity,'特异率':Specificity,'准确率':Accuracy,
                '曲线下面积':AUC,'CIlower':CI_lower,'CIupper':CI_upper,'约旦指数':Youden,'区分阈值':Optimal}
    df= pd.DataFrame(dataoutput)
    df['曲线下的面积']=df.apply(lambda x:str(x['曲线下面积'])+"("+str(x['CIlower'])+"-"+str(x['CIupper'])+")",axis=1)
    df.to_csv("C:/Users/mjdee/Desktop/JI-2020/ML/simple.csv",encoding='GBK')
    predict_prob=pd.DataFrame(predict_prob)
    predict_prob['真实死活']=y_test
    predict_prob['姓名']=patient.iloc[:,0:1]
    predict_prob['id']=patient.iloc[:,1:2]
    predict_prob.to_csv(r"C:\Users\mjdee\Desktop\JI-2020\ML\predicted.csv",encoding='GBK')
