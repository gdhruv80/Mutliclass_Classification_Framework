
# coding: utf-8

# In[1]:


get_ipython().run_cell_magic('javascript', '', 'IPython.OutputArea.auto_scroll_threshold = 9999;')


# In[2]:


# Sklearn 
from sklearn.datasets import *
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier, BayesianRidge
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
from scipy.stats import randint as rint, uniform as runi
from numpy.random import uniform as uni, randint as unint
from sklearn.metrics import roc_curve,roc_auc_score,accuracy_score,confusion_matrix,precision_recall_curve
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import numpy as np
import pandas as pd
import sys 
import matplotlib.pyplot as plt

# Personal Classification repo
from Garg_Classif_hyper import report_top_hyp,param_grid,models, feature_selection,hyperparam_search,imputers
from Garg_Classif_roc import ROC_operating_point , ROC_optim_oper , final_mod_perf
from Garg_Classif_plotting import ROC_plot , Precision_Recall_plot,outlier_analysis
from Garg_Preprocessing import train_test_dev_split , outlier_det

# Quick useful numpy functions
roundnp = np.vectorize(lambda t: round(t,3))

# Notebook Output viewing settings 
import warnings
import math
warnings.filterwarnings('ignore')


# In[3]:


# Plot distribution of all independent variables
# Before inputting your dataset into dat_preprocess make sure no date time vars are there
# Convert to numeric do feature engineering before proceeding further
breast = load_breast_cancer()
dat = breast.data
tar = breast.target
df = pd.DataFrame(data = dat,columns = breast.feature_names)
tar = pd.DataFrame(data = tar,columns = ['outcome'])
df = pd.concat((df,tar),axis = 1)
df['cat1'] = np.where(df['mean radius']>17,'hi','bye')
df['cat2'] = df.apply(lambda x: 'to' if x['mean radius'] < 13 
                      else np.nan if x['mean radius'] < 18 else 'no',axis = 1)
df.iloc[0:25,0:2] = np.nan
#df['date'] = pd.to_datetime('19000101', format='%Y%m%d', errors='ignore')


# In[4]:


def dat_preprocess(df,outcome_var,alow_miss_X_prcnt,*args):
    """
    df - Complete DF including outcome var
    args are the categorical predictors for one hot encoding specify in list format [v1,v2,..]
    outcome_var = Just the name of the outcome var - eg : 'No_of_tickets'
    """
    # One hot encoding categorical variables and output variable
    try:
        if args[0]:
            data_set = pd.get_dummies(df,columns = args[0])
    except:
        print('No categorical predictors for one hot encoding provided')
        data_set = df
    
    # Removing rows for Outcome variable with NAN values
    print ('No of rows with outcome variable not populated : %d' 
           %(data_set[outcome_var].isna().sum()))
           
    data_set = data_set.loc[~data_set[outcome_var].isna(),:]
    
    print ('Orignal row count : %d' %(len(data_set)))
    # Removing any duplicate rows
    data_set.drop_duplicates(inplace = True)
    print ('Deduplicated row count : %d' %(len(data_set)))
    
    #Checking and removing any predictors with >X% of the values as missing
    per_miss_pred = ((data_set.isna().sum(axis = 0)/len(data_set))*100).round(1)
    # Printing % missing value count if missing values > 0 %
    print ('Variables with some missing values (ie > 0 percent):')
    print (per_miss_pred.loc[per_miss_pred > 0])
    if len(per_miss_pred.loc[per_miss_pred > alow_miss_X_prcnt]) == 0:
        print ('No predictors with missing percent more than %d' %(alow_miss_X_prcnt))
    else:
        print('Removing the following predictors               as they have more than %d values missing' %(alow_miss_X_prcnt))
        print(list(per_miss_pred.loc[per_miss_pred > alow_miss_X_prcnt].index))
    
    data_set_mod = data_set[per_miss_pred.loc[per_miss_pred <= alow_miss_X_prcnt].index]
        
    # Label encoding the Y variable
    le = LabelEncoder()
    data_set_mod[outcome_var] = le.fit_transform(data_set_mod[outcome_var])
    print (' ')
    print ('Encoding for the outcome variable')
    le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    print(le_name_mapping)
    
    # Splitting into train test and dev
    X_train, X_dev, X_test, Y_train, Y_dev, Y_test =     train_test_dev_split(data_set_mod, outcome_var)
    
    return X_train, X_dev, X_test, Y_train, Y_dev, Y_test


# In[5]:


def make_train_pipe(feat_selec,mod_class,imput_model,hyper_selc,rand_search_iter,scoring_crit,
                    X_train,Y_train,X_dev,X_test,random_state):
    """
    imput_model: Itterative imputation method model - BaysRidge(bays),RFregressor(Rf),Knn(Knn)
    mod_class: either Tree_based , Reg_based
    feat_selec: either KBest, L1 or PCA
    hyper_selc: either Grid or Random
    scoring_crit: Scoring criteria for top model f1,roc_auc,balanced_accuracy,precision,recall
    """
    scale_data = StandardScaler()
    imputation = imputers(imput_model,random_state)
    feat_selection_ = feature_selection(feat_selec)
    mod_ = models(mod_class,random_state)
    
    hyp_search_res = {}
    for mod_type,mod in mod_.items():
        pipe = Pipeline(steps=[('scaled_data',scale_data),
                               (imput_model,imputation),
                               (feat_selec, feat_selection_), 
                               (mod_type, mod)])
        
        param = param_grid(feat_selec,mod_type,hyper_selc,rand_search_iter)
        
        search = hyperparam_search(hyper_selc,pipe,param,rand_search_iter,scoring_crit)
        
        search.fit(X_train, Y_train)
        # Predict the prob of positive class
        pred_dev = search.predict_proba(X_dev)[:,list(search.classes_).index(1)] 
        pred_test = search.predict_proba(X_test)[:,list(search.classes_).index(1)] 

        print ('MODEL TYPE : %s' %(mod_type.upper()))
        print ('')
        report_top_hyp(search.cv_results_,scoring_crit)
        print ('-----------------      ------------------')
        print ('')
        hyp_search_res[mod_type] = (search.best_params_,pred_dev,pred_test)
    
    return hyp_search_res


# In[6]:


def tune_roc(Y_pred_prob,Y_act,fn_cost,fp_cost,Y_test_pred_prob):
    """
    Y_pred_prob , Y_act : Pred prob and labels of the Dev set (from hyp param tuning step)
    Y_test_pred_prob :  Pred prob and labels of the Test set (from hyp param tuning step) 
    Moves operating point up or down based on seperate costs of FP and FN and plot new ROC
    Also plots precision recall plot for the new operating point
    """
    random_pred = np.random.randint(0,2,len(Y_act))
    fpr,tpr,thresholds = roc_curve(Y_act,Y_pred_prob,pos_label=1)
    fpr_ran,tpr_ran,thresholds_ran = roc_curve(Y_act,random_pred,pos_label=1)
    # Plotting best model and random classifier
    print ('--------------')
    print ('EVALUATING PERFORMANCE ON DEV SET')
    print (' ')
    print ("The AUC area of the Best classifier is: %.2f" %(roc_auc_score(Y_act,Y_pred_prob)))
    print ("The AUC area of a Random classifier is: %.2f" %(roc_auc_score(Y_act,random_pred)))
    
    # Marking the DEFAULT operation point for the selected model - GBC    
    AUC_char = pd.DataFrame(data = np.array([fpr,tpr,thresholds]).T,
                     columns = ['Fpr','Tpr','Thresholds'])
    
    oper_fpr, oper_tpr = ROC_operating_point(AUC_char, 0.5)['Fpr'],     ROC_operating_point(AUC_char, 0.5)['Tpr']
    
    # Optimizing for the operating point 
    opt_thres,(opt_cost,opt_tpr,opt_fpr,def_cost) = ROC_optim_oper(Y_pred_prob,Y_act,fn_cost,fp_cost )
    
    # Plotting the ROC 
    plt.figure(figsize=(8, 12))
    ROC_plot(fpr,tpr,oper_fpr,oper_tpr,opt_fpr,opt_tpr,fpr_ran,tpr_ran)
    
    # Plotting Precision Recall 
    Precision_Recall_plot(Y_act,Y_pred_prob,opt_thres)
    
    print('The new operating point is %.2f compared to the default of 0.50.' %(opt_thres))
    print('This has reduced the overall cost of misclassification from %d to %d' %(def_cost,opt_cost))
    f1 = np.vectorize(lambda x :1 if x > opt_thres else 0)
    return f1(Y_test_pred_prob)


# In[7]:


X_train,X_dev, X_test,Y_train,Y_dev,Y_test = dat_preprocess(df,'outcome',30,['cat1','cat2'])

X_train_nooutlier,Y_train_nooutlier = outlier_det(X_train,Y_train,'texture error','mean compactness','Robust covariance',0.05)   

outlier_analysis(X_train,X_train_nooutlier)

Train_res = make_train_pipe('L1','Tree_based','bays','Random',10,'f1', 
                            X_train_nooutlier,Y_train_nooutlier,X_dev,X_test,123)

tuned_pred = tune_roc(Train_res['rf'][1],Y_dev,1,5,Train_res['rf'][2])
        
final_mod_perf(tuned_pred,Y_test,no_output_class = 2)

