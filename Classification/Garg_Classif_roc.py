import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


# Utility function to extract operating point for ROC curve
def ROC_operating_point(df, prob):
    """
    :param df: Dataframe in format Fpr ,Tpr ,Thresholds
    :param prob: Operating probability
    :returns Dict of mean Fpr and Tpr for the operating threshold probability specified as input
    """
    df = df.round(2)
    if len(df.loc[df['Thresholds'] == 0.5]) > 0:
        return {'Fpr': df.loc[df['Thresholds'] == 0.5, 'Fpr'].mean(),
                'Tpr': df.loc[df['Thresholds'] == 0.5, 'Tpr'].mean()}
    else:
        for i in range(len(df)):
            if df.iloc[i, 2] < prob: break
        return {'Fpr': df.iloc[i - 1:i + 1, 0].mean(), 'Tpr': df.iloc[i - 1:i + 1, 1].mean()}


# Utility function to optimize for new Operating Point
def ROC_optim_oper(Y_pred_prob, Y_act, fn_cost=1, fp_cost=1):
    """
    :returns A tuple of type (New_threshold, (new_cost, TPR, FPR, Default_cost))
    """
    # Default calculation
    tn_def, fp_def, fn_def, tp_def = confusion_matrix(Y_act, np.where(Y_pred_prob > 0.5, 1, 0)).ravel()
    def_cost = fn_def * fn_cost + fp_def * fp_cost

    # Optimized calculation
    df = pd.DataFrame(data=np.array([Y_pred_prob, Y_act]).T, columns=['Y_pred_prob', 'Y_'])
    df = df.round(2)
    uniq_thresh = list(df['Y_pred_prob'].unique())
    thres_cost = {}
    for i in uniq_thresh:
        Y_pred_bnary = df['Y_pred_prob'].apply(lambda x: 1 if x > i else 0)
        tn, fp, fn, tp = confusion_matrix(df['Y_'], Y_pred_bnary).ravel()
        thres_cost[i] = tuple(map(lambda x: round(x, 2),
                                  (fn * fn_cost + fp * fp_cost, tp / (tp + fn), fp / (fp + tn), def_cost)))

    return sorted(thres_cost.items(), key=lambda x: x[1][0], reverse=False)[0]


def final_mod_perf(Y_pred, Y_act, no_output_class=2):
    """
    :param Y_pred: Y predicted value from classifier (Absolute class not probability) Can be multi class as well
    :param Y_act: Actual labels
    :param no_output_class: No of output classes in the output label
    :return: Nothing
    """
    Y_rand_pred = np.random.randint(0, no_output_class, len(Y_pred))
    if no_output_class == 2:
        tn, fp, fn, tp = confusion_matrix(Y_act, Y_pred).ravel()
        tn_def, fp_def, fn_def, tp_def = confusion_matrix(Y_act, Y_rand_pred).ravel()
    else:
        tn, fp, fn, tp = multi_class_tn_fp(confusion_matrix(Y_act, Y_pred))
        tn_def, fp_def, fn_def, tp_def = multi_class_tn_fp(confusion_matrix(Y_act, Y_rand_pred))

    # print(tn, fp, fn, tp)
    tnr, fpr, fnr, tpr = get_tpr_4(tn, fp, fn, tp)
    tnr_def, fpr_def, fnr_def, tpr_def = get_tpr_4(tn_def, fp_def, fn_def, tp_def)
    # print(tnr, fpr, fnr, tpr)
    # Averaging tpr, fpr etc for multi class case across all classes
    if no_output_class != 2:
        tnr, fpr, fnr, tpr, tnr_def, fpr_def, fnr_def, tpr_def = map(lambda x: np.mean(x), (
        tnr, fpr, fnr, tpr, tnr_def, fpr_def, fnr_def, tpr_def))
    print('  ')
    print('--------------')
    print('PERFORMANCE ON UNTOUCHED TEST SET')
    print(
        'Performance of the post tuned classifier (Both hyperparameter and operating point) compared to the default '
        'classifier (ie if we randomly predicted positive/negetive based on a fair coin toss) :')
    print(
        'If there are 100 +ve cases our model is able to identify %s of all the positive cases correctly while only '
        'misclassifying %s of those cases as negative' % (
        round(tpr * 100, 2), round(fnr * 100, 2)))
    print(
        'While on the other hand the random classifier is only able to identify %s of all the positive cases '
        'correctly ' % (
            round(tpr_def * 100, 2)))
    print(' ')
    print(
        'This is a LIFT of %s times in predicting the positive class over the performance of the default classifier' % (
            round(tpr / tpr_def, 1)))
    print(' ')
    print(
        'Similarly if there are 100 -ve cases our model is able to identify %s of all the negative cases correctly '
        'while costing us very little by only misclassifying %s of those cases as positive' % (
        round(tnr * 100, 2), round(fpr * 100, 2)))
    print(
        'While on the other hand the random classifier costs us a lot (by misclasifying -ve as +ve) trying to predict '
        'the postive cases and misclassifies %s of all the negative cases as positive' % (
            round(fpr_def * 100, 2)))


def get_tpr_4(tn, fp, fn, tp):
    """
    :param tn: TN
    :param fp: FP
    :param fn: FN
    :param tp: TP
    :return: All 4 TNR, FPR, FNR, TPR
    """
    return tn / (tn + fp), fp / (fp + tn), fn / (fn + tp), tp / (tp + fn)


def multi_class_tn_fp(confusion_matrix):
    """
    :param confusion_matrix: Take in a multi class >2 confusion matrix
    :return: tn,fp,fn,tp values
    """
    fp = confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)
    fn = confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)
    tp = np.diag(confusion_matrix)
    tn = confusion_matrix.sum() - (fp + fn + tp)
    return tn, fp, fn, tp
