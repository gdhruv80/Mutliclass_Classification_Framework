import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve


def outlier_analysis(df_bef,df_aft):
    """
    :param df_bef: Pre outlier detection DF
    :param df_aft: Post outlier detection DF
    :return: None (Just Plots for each variable upto the first 18 variables)
    """
    plt.figure(figsize=(20, 36))
    # Subsetting only numeric variables
    df_bef_ = df_bef[df_bef.dtypes[df_bef.dtypes.isin([np.dtype('float64'),
                                                       np.dtype('int64')])].index]
    df_aft_ = df_aft[df_aft.dtypes[df_aft.dtypes.isin([np.dtype('float64'),
                                                       np.dtype('int64')])].index]
    # Plot the first 18 variables
    n_cols = 18 if len(df_bef_.columns) > 18 else len(df_bef_.columns)
    sub_len  = math.ceil(n_cols/3)
    for i in range(n_cols):
        plt.subplot(sub_len,3,i+1)
        p1 = df_bef.iloc[:,i].plot.kde(color = 'red',title = df_bef.columns[i].upper(),
                                       legend = True,linewidth=3)
        p2 = df_aft.iloc[:,i].plot.kde(color = 'blue',linewidth=2)
        p2.legend(["PDE - Orignal","PDE - Post outlier Detection"])
        p2.set_xlabel(df_bef.columns[i])

#Plotting the ROC
def ROC_plot(fpr,tpr,oper_fpr,oper_tpr,opt_fpr,opt_tpr,fpr_ran,tpr_ran):
    """
    :param fpr: 
    :param tpr: 
    :param oper_fpr: 
    :param oper_tpr: 
    :param opt_fpr: 
    :param opt_tpr: 
    :param fpr_ran: 
    :param tpr_ran: 
    :return: 
    """
    plt.subplot(211)
    plt.plot(fpr,tpr,lw=2,label='ROC_ChosenModel')
    plt.plot(fpr_ran,tpr_ran,lw=2,label='ROC_Random_Classifier',linestyle = '--',color = 'black')
    plt.plot(oper_fpr,oper_tpr,'ro',color = 'red',label= 'Default operation point')
    plt.plot(opt_fpr,opt_tpr,'ro',color = 'green',label= 'New Optimized operation point')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic graph')
    plt.legend(loc="lower right")

def Precision_Recall_plot(Y_act,Y_pred_prob,opt_thres):
    """
    :param Y_act:
    :param Y_pred_prob:
    :param opt_thres:
    :return:
    """
    plt.subplot(212)
    precision, recall, thresholds = precision_recall_curve(Y_act, Y_pred_prob)
    plt.plot(thresholds, precision[:-1], "b--", label="Precision")
    plt.plot(thresholds, recall[:-1], "g-", label="Recall")
    plt.ylabel("Precision/Recall Value")
    plt.xlabel("Decision Threshold")
    plt.axvline(x=opt_thres, linestyle=':', color='peachpuff', label='New Optimized Operating Point')
    plt.axvline(x=0.5, color='lightgrey', linestyle=':', label='Default Operating Point')
    plt.title("Precision and Recall Scores as a function of the decision threshold")
    plt.legend(loc='best')


