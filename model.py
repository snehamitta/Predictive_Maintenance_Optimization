import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import sklearn.cluster as cluster
import sklearn.decomposition as decomposition
import sklearn.metrics as metrics
import sklearn.linear_model as linear_model
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import sklearn.metrics as metrics
import statsmodels.api as api
from sklearn.metrics import zero_one_loss
from sklearn.metrics import mean_squared_error
from math import sqrt

data = pd.read_csv('/Users/snehamitta/Desktop/ML/Final/fleet_train.csv', delimiter = ',')

col = data.columns

Xtrain = data.iloc[:,1:]
Ytrain = data[['Maintenance_flag','Region']]

Xtrain.drop([col[1]], axis=1, inplace = True)
Xtrain.drop([col[2]], axis=1, inplace = True)
Xtrain.drop([col[28]], axis=1, inplace = True)
Xtrain.drop([col[29]], axis=1, inplace = True)

# Region 1

XtrainR1 = Xtrain.loc[Xtrain['Region'] == 1]
XtrainR1.drop([col[30]], axis= 1, inplace = True)

YtrainR1 = Ytrain.loc[Ytrain['Region'] == 1]
YtrainR1.drop([col[30]], axis = 1, inplace = True)

lasso = Lasso(alpha=0.3)
lasso.fit(XtrainR1, YtrainR1)
print ('Region 1: ',lasso.coef_)

Ynew1 = YtrainR1['Maintenance_flag'].astype('category')
y = Ynew1
y_category = y.cat.categories

XtrainR1.drop([col[4]], axis= 1, inplace = True)
XtrainR1.drop([col[5]], axis= 1, inplace = True)
XtrainR1.drop([col[6]], axis= 1, inplace = True)
XtrainR1.drop([col[7]], axis= 1, inplace = True)
XtrainR1.drop([col[10]], axis= 1, inplace = True)
XtrainR1.drop([col[11]], axis= 1, inplace = True)
XtrainR1.drop([col[12]], axis= 1, inplace = True)
XtrainR1.drop([col[13]], axis= 1, inplace = True)
XtrainR1.drop([col[14]], axis= 1, inplace = True)
XtrainR1.drop([col[15]], axis= 1, inplace = True)
XtrainR1.drop([col[16]], axis= 1, inplace = True)
XtrainR1.drop([col[18]], axis= 1, inplace = True)
XtrainR1.drop([col[19]], axis= 1, inplace = True)
XtrainR1.drop([col[22]], axis= 1, inplace = True)
XtrainR1.drop([col[23]], axis= 1, inplace = True)
XtrainR1.drop([col[24]], axis= 1, inplace = True)
XtrainR1.drop([col[25]], axis= 1, inplace = True)
XtrainR1.drop([col[26]], axis= 1, inplace = True)

testdata = pd.read_csv('/Users/snehamitta/Desktop/ML/Final/fleet_monitor_notscored_2.csv', delimiter = ',')

Xtest = testdata.iloc[:,1:]
Ytest = testdata[['Maintenance_flag','Region']]

col = testdata.columns

Xtest.drop([col[1]], axis=1, inplace = True)
Xtest.drop([col[2]], axis=1, inplace = True)
Xtest.drop([col[28]], axis=1, inplace = True)
Xtest.drop([col[29]], axis=1, inplace = True)
Xtest.drop([col[30]], axis=1, inplace = True)

XtestR1 = Xtest.loc[Xtest['Region'] == 1]
XtestR1.drop([col[31]], axis= 1, inplace = True)

YtestR1 = Ytest.loc[Ytest['Region'] == 1]
YtestR1.drop([col[31]], axis = 1, inplace = True)

XtestR1.drop([col[4]], axis= 1, inplace = True)
XtestR1.drop([col[5]], axis= 1, inplace = True)
XtestR1.drop([col[6]], axis= 1, inplace = True)
XtestR1.drop([col[7]], axis= 1, inplace = True)
XtestR1.drop([col[10]], axis= 1, inplace = True)
XtestR1.drop([col[11]], axis= 1, inplace = True)
XtestR1.drop([col[12]], axis= 1, inplace = True)
XtestR1.drop([col[13]], axis= 1, inplace = True)
XtestR1.drop([col[14]], axis= 1, inplace = True)
XtestR1.drop([col[15]], axis= 1, inplace = True)
XtestR1.drop([col[16]], axis= 1, inplace = True)
XtestR1.drop([col[18]], axis= 1, inplace = True)
XtestR1.drop([col[19]], axis= 1, inplace = True)
XtestR1.drop([col[22]], axis= 1, inplace = True)
XtestR1.drop([col[23]], axis= 1, inplace = True)
XtestR1.drop([col[24]], axis= 1, inplace = True)
XtestR1.drop([col[25]], axis= 1, inplace = True)
XtestR1.drop([col[26]], axis= 1, inplace = True)

logit = api.MNLogit(YtrainR1, XtrainR1)
print("Name of Target Variable:", logit.endog_names)
print("Name(s) of Predictors:", logit.exog_names)

thisFit = logit.fit(method='newton', full_output = True, maxiter = 100, tol = 1e-8)
thisParameter = thisFit.params

print("Model Parameter Estimates:\n", thisFit.params)
print("Model Log-Likelihood Value:\n", logit.loglike(thisParameter.values))

y_predProbR1 = thisFit.predict(XtestR1)
print(y_predProbR1.mean())
y_predictR1 = pd.to_numeric(y_predProbR1.idxmax(axis=1))

y_predictClass = y_category[y_predictR1]
y_accuracy = metrics.accuracy_score(YtestR1, y_predictClass)
print("Accuracy Score = ", y_accuracy)

y_confusion = metrics.confusion_matrix(YtestR1, y_predictClass)
print("Confusion Matrix (Row is True, Column is Predicted) = \n")
print(y_confusion)

# Region 2

data = pd.read_csv('/Users/snehamitta/Desktop/ML/Final/fleet_train.csv', delimiter = ',')

col = data.columns

Xtrain = data.iloc[:,1:]
Ytrain = data[['Maintenance_flag','Region']]

Xtrain.drop([col[1]], axis=1, inplace = True)
Xtrain.drop([col[2]], axis=1, inplace = True)
Xtrain.drop([col[28]], axis=1, inplace = True)
Xtrain.drop([col[29]], axis=1, inplace = True)

XtrainR2 = Xtrain.loc[Xtrain['Region'] == 2]
XtrainR2.drop([col[30]], axis= 1, inplace = True)

YtrainR2 = Ytrain.loc[Ytrain['Region'] == 2]
YtrainR2.drop([col[30]], axis = 1, inplace = True)

lasso = Lasso(alpha=0.3)
lasso.fit(XtrainR2, YtrainR2)
print ('Region 2', lasso.coef_)

Ynew2 = YtrainR2['Maintenance_flag'].astype('category')
y = Ynew2
y_category = y.cat.categories

XtrainR2.drop([col[3]], axis= 1, inplace = True)
XtrainR2.drop([col[4]], axis= 1, inplace = True)
XtrainR2.drop([col[6]], axis= 1, inplace = True)
XtrainR2.drop([col[7]], axis= 1, inplace = True)
XtrainR2.drop([col[9]], axis= 1, inplace = True)
XtrainR2.drop([col[10]], axis= 1, inplace = True)
XtrainR2.drop([col[11]], axis= 1, inplace = True)
XtrainR2.drop([col[12]], axis= 1, inplace = True)
XtrainR2.drop([col[13]], axis= 1, inplace = True)
XtrainR2.drop([col[14]], axis= 1, inplace = True)
XtrainR2.drop([col[15]], axis= 1, inplace = True)
XtrainR2.drop([col[16]], axis= 1, inplace = True)
XtrainR2.drop([col[17]], axis= 1, inplace = True)
XtrainR2.drop([col[18]], axis= 1, inplace = True)
XtrainR2.drop([col[19]], axis= 1, inplace = True)
XtrainR2.drop([col[22]], axis= 1, inplace = True)
XtrainR2.drop([col[23]], axis= 1, inplace = True)
XtrainR2.drop([col[24]], axis= 1, inplace = True)
XtrainR2.drop([col[25]], axis= 1, inplace = True)

testdata = pd.read_csv('/Users/snehamitta/Desktop/ML/Final/fleet_monitor_notscored_2.csv', delimiter = ',')

Xtest = testdata.iloc[:,1:]
Ytest = testdata[['Maintenance_flag','Region']]

col = testdata.columns

Xtest.drop([col[1]], axis=1, inplace = True)
Xtest.drop([col[2]], axis=1, inplace = True)
Xtest.drop([col[28]], axis=1, inplace = True)
Xtest.drop([col[29]], axis=1, inplace = True)
Xtest.drop([col[30]], axis=1, inplace = True)

XtestR2 = Xtest.loc[Xtest['Region'] == 2]
XtestR2.drop([col[31]], axis= 1, inplace = True)

YtestR2 = Ytest.loc[Ytest['Region'] == 2]
YtestR2.drop([col[31]], axis = 1, inplace = True)

XtestR2.drop([col[3]], axis= 1, inplace = True)
XtestR2.drop([col[4]], axis= 1, inplace = True)
XtestR2.drop([col[6]], axis= 1, inplace = True)
XtestR2.drop([col[7]], axis= 1, inplace = True)
XtestR2.drop([col[9]], axis= 1, inplace = True)
XtestR2.drop([col[10]], axis= 1, inplace = True)
XtestR2.drop([col[11]], axis= 1, inplace = True)
XtestR2.drop([col[12]], axis= 1, inplace = True)
XtestR2.drop([col[13]], axis= 1, inplace = True)
XtestR2.drop([col[14]], axis= 1, inplace = True)
XtestR2.drop([col[15]], axis= 1, inplace = True)
XtestR2.drop([col[16]], axis= 1, inplace = True)
XtestR2.drop([col[17]], axis= 1, inplace = True)
XtestR2.drop([col[18]], axis= 1, inplace = True)
XtestR2.drop([col[19]], axis= 1, inplace = True)
XtestR2.drop([col[22]], axis= 1, inplace = True)
XtestR2.drop([col[23]], axis= 1, inplace = True)
XtestR2.drop([col[24]], axis= 1, inplace = True)
XtestR2.drop([col[25]], axis= 1, inplace = True)

logit = api.MNLogit(YtrainR2, XtrainR2)
print("Name of Target Variable:", logit.endog_names)
print("Name(s) of Predictors:", logit.exog_names)

thisFit = logit.fit(method='newton', full_output = True, maxiter = 100, tol = 1e-8)
thisParameter = thisFit.params

print("Model Parameter Estimates:\n", thisFit.params)
print("Model Log-Likelihood Value:\n", logit.loglike(thisParameter.values))

y_predProbR2 = thisFit.predict(XtestR2)
print(y_predProbR2.mean())
y_predictR2 = pd.to_numeric(y_predProbR2.idxmax(axis=1))

y_predictClass = y_category[y_predictR2]
y_accuracy = metrics.accuracy_score(YtestR2, y_predictClass)
print("Accuracy Score = ", y_accuracy)

y_confusion = metrics.confusion_matrix(YtestR2, y_predictClass)
print("Confusion Matrix (Row is True, Column is Predicted) = \n")
print(y_confusion)

# Region 3

data = pd.read_csv('/Users/snehamitta/Desktop/ML/Final/fleet_train.csv', delimiter = ',')

col = data.columns

Xtrain = data.iloc[:,1:]
Ytrain = data[['Maintenance_flag','Region']]

Xtrain.drop([col[1]], axis=1, inplace = True)
Xtrain.drop([col[2]], axis=1, inplace = True)
Xtrain.drop([col[28]], axis=1, inplace = True)
Xtrain.drop([col[29]], axis=1, inplace = True)

XtrainR3 = Xtrain.loc[Xtrain['Region'] == 3]
XtrainR3.drop([col[30]], axis= 1, inplace = True)

YtrainR3 = Ytrain.loc[Ytrain['Region'] == 3]
YtrainR3.drop([col[30]], axis = 1, inplace = True)

lasso = Lasso(alpha=0.3)

lasso.fit(XtrainR3, YtrainR3)
print ('Region 3', lasso.coef_)

Ynew3 = YtrainR3['Maintenance_flag'].astype('category')
y = Ynew3
y_category = y.cat.categories

XtrainR3.drop([col[3]], axis= 1, inplace = True)
XtrainR3.drop([col[4]], axis= 1, inplace = True)
XtrainR3.drop([col[6]], axis= 1, inplace = True)
XtrainR3.drop([col[7]], axis= 1, inplace = True)
XtrainR3.drop([col[9]], axis= 1, inplace = True)
XtrainR3.drop([col[10]], axis= 1, inplace = True)
XtrainR3.drop([col[11]], axis= 1, inplace = True)
XtrainR3.drop([col[12]], axis= 1, inplace = True)
XtrainR3.drop([col[13]], axis= 1, inplace = True)
XtrainR3.drop([col[14]], axis= 1, inplace = True)
XtrainR3.drop([col[15]], axis= 1, inplace = True)
XtrainR3.drop([col[16]], axis= 1, inplace = True)
XtrainR3.drop([col[18]], axis= 1, inplace = True)
XtrainR3.drop([col[19]], axis= 1, inplace = True)
XtrainR3.drop([col[22]], axis= 1, inplace = True)
XtrainR3.drop([col[23]], axis= 1, inplace = True)
XtrainR3.drop([col[24]], axis= 1, inplace = True)
XtrainR3.drop([col[25]], axis= 1, inplace = True)

testdata = pd.read_csv('/Users/snehamitta/Desktop/ML/Final/fleet_monitor_notscored_2.csv', delimiter = ',')

Xtest = testdata.iloc[:,1:]
Ytest = testdata[['Maintenance_flag','Region']]

col = testdata.columns

Xtest.drop([col[1]], axis=1, inplace = True)
Xtest.drop([col[2]], axis=1, inplace = True)
Xtest.drop([col[28]], axis=1, inplace = True)
Xtest.drop([col[29]], axis=1, inplace = True)
Xtest.drop([col[30]], axis=1, inplace = True)

XtestR3 = Xtest.loc[Xtest['Region'] == 3]
XtestR3.drop([col[31]], axis= 1, inplace = True)

YtestR3 = Ytest.loc[Ytest['Region'] == 3]
YtestR3.drop([col[31]], axis = 1, inplace = True)

XtestR3.drop([col[3]], axis= 1, inplace = True)
XtestR3.drop([col[4]], axis= 1, inplace = True)
XtestR3.drop([col[6]], axis= 1, inplace = True)
XtestR3.drop([col[7]], axis= 1, inplace = True)
XtestR3.drop([col[9]], axis= 1, inplace = True)
XtestR3.drop([col[10]], axis= 1, inplace = True)
XtestR3.drop([col[11]], axis= 1, inplace = True)
XtestR3.drop([col[12]], axis= 1, inplace = True)
XtestR3.drop([col[13]], axis= 1, inplace = True)
XtestR3.drop([col[14]], axis= 1, inplace = True)
XtestR3.drop([col[15]], axis= 1, inplace = True)
XtestR3.drop([col[16]], axis= 1, inplace = True)
XtestR3.drop([col[18]], axis= 1, inplace = True)
XtestR3.drop([col[19]], axis= 1, inplace = True)
XtestR3.drop([col[22]], axis= 1, inplace = True)
XtestR3.drop([col[23]], axis= 1, inplace = True)
XtestR3.drop([col[24]], axis= 1, inplace = True)
XtestR3.drop([col[25]], axis= 1, inplace = True)

logit = api.MNLogit(YtrainR3, XtrainR3)
print("Name of Target Variable:", logit.endog_names)
print("Name(s) of Predictors:", logit.exog_names)

thisFit = logit.fit(method='newton', full_output = True, maxiter = 100, tol = 1e-8)
thisParameter = thisFit.params

print("Model Parameter Estimates:\n", thisFit.params)
print("Model Log-Likelihood Value:\n", logit.loglike(thisParameter.values))

y_predProbR3 = thisFit.predict(XtestR3)
print(y_predProbR3.mean())
y_predictR3 = pd.to_numeric(y_predProbR3.idxmax(axis=1))

y_predictClass = y_category[y_predictR3]
y_accuracy = metrics.accuracy_score(YtestR3, y_predictClass)
print("Accuracy Score = ", y_accuracy)

y_confusion = metrics.confusion_matrix(YtestR3, y_predictClass)
print("Confusion Matrix (Row is True, Column is Predicted) = \n")
print(y_confusion)

#Comparison of predicted probabilities

p = [y_predProbR1,y_predProbR2,y_predProbR3]
q = [YtestR1, YtestR2, YtestR3]

predprob = pd.concat(p)
predprob_flags = predprob.iloc[:,1]
predprob_flags.sort_index()
y_test = pd.concat(q)

# Model Evaluation metrics

fpr, tpr, thresholds = metrics.roc_curve(y_test, predprob_flags)
print('The AUC is for Logistic model is', metrics.auc(fpr, tpr))

f = []
for i in (predprob_flags):
    if i >= 0.20469083:
        f.append(1)
    else:
        f.append(0)

error_rate = zero_one_loss(y_test, f)
print('The misclassification rate of the Logistic model is', error_rate)

rms = sqrt(mean_squared_error(y_test, predprob_flags))
print('The RASE value for Logistic model is', rms)

plt.plot(fpr, tpr, color = 'blue', linestyle = 'solid', linewidth = 2, label = 'Logistic Regression')
plt.plot([0, 1], [0, 1], color = 'red', linestyle = ':', label = 'Ref. Line')
plt.xlabel("1 - Specificity")
plt.ylabel("Sensitivity")
plt.legend(loc = 'upper left')
plt.title('Receiver Operating Characteristic curve')
plt.grid(True)
plt.show()

score_test_lr = pd.concat([y_test, predprob], axis = 1)

def compute_lift_coordinates (
        DepVar,          # The column that holds the dependent variable's values
        EventValue,      # Value of the dependent variable that indicates an event
        EventPredProb,   # The column that holds the predicted event probability
        Debug = 'N'):    # Show debugging information (Y/N)

    # Find out the number of observations
    nObs = len(DepVar)

    # Get the quantiles
    quantileCutOff = np.percentile(EventPredProb, np.arange(0, 100, 10))
    nQuantile = len(quantileCutOff)

    quantileIndex = np.zeros(nObs)
    for i in range(nObs):
        iQ = nQuantile
        EPP = EventPredProb.iloc[i]
        for j in range(1, nQuantile):
            if (EPP > quantileCutOff[-j]):
                iQ -= 1
        quantileIndex[i] = iQ

    # Construct the Lift chart table
    countTable = pd.crosstab(quantileIndex, DepVar)
    decileN = countTable.sum(1)
    decilePct = 100 * (decileN / nObs)
    gainN = countTable[EventValue]
    totalNResponse = gainN.sum(0)
    gainPct = 100 * (gainN /totalNResponse)
    responsePct = 100 * (gainN / decileN)
    overallResponsePct = 100 * (totalNResponse / nObs)
    lift = responsePct / overallResponsePct

    LiftCoordinates = pd.concat([decileN, decilePct, gainN, gainPct, responsePct, lift],
                                    axis = 1, ignore_index = True)
    LiftCoordinates = LiftCoordinates.rename({0:'Decile N',
                                              1:'Decile %',
                                              2:'Gain N',
                                              3:'Gain %',
                                              4:'Response %',
                                              5:'Lift'}, axis = 'columns')

    # Construct the Accumulative Lift chart table
    accCountTable = countTable.cumsum(axis = 0)
    decileN = accCountTable.sum(1)
    decilePct = 100 * (decileN / nObs)
    gainN = accCountTable[EventValue]
    gainPct = 100 * (gainN / totalNResponse)
    responsePct = 100 * (gainN / decileN)
    lift = responsePct / overallResponsePct

    accLiftCoordinates = pd.concat([decileN, decilePct, gainN, gainPct, responsePct, lift],
                                       axis = 1, ignore_index = True)
    accLiftCoordinates = accLiftCoordinates.rename({0:'Acc. Decile N',
                                                    1:'Acc. Decile %',
                                                    2:'Acc. Gain N',
                                                    3:'Acc. Gain %',
                                                    4:'Acc. Response %',
                                                    5:'Acc. Lift'}, axis = 'columns')
        
    if (Debug == 'Y'):
        print('Number of Quantiles = ', nQuantile)
        print(quantileCutOff)
        _u_, _c_ = np.unique(quantileIndex, return_counts = True)
        print('Quantile Index: \n', _u_)
        print('N Observations per Quantile Index: \n', _c_)
        print('Count Table: \n', countTable)
        print('Accumulated Count Table: \n', accCountTable)

    return(LiftCoordinates, accLiftCoordinates)

lift_coordinates, acc_lift_coordinates = compute_lift_coordinates (
        DepVar = score_test_lr['Maintenance_flag'],
        EventValue = 1,
        EventPredProb = score_test_lr[1],
        Debug = 'Y')

lift_lr = lift_coordinates
acc_lr = acc_lift_coordinates


plt.plot(acc_lr.index, acc_lr['Acc. Lift'], marker = 'o',
         color = 'purple', linestyle = 'solid', linewidth = 2, markersize = 6, label = 'Logistic Regression')

plt.title("Testing Partition Acc. Lift")
plt.grid(True)
plt.xticks(np.arange(1,11, 1))
plt.xlabel("Decile Group")
plt.ylabel("Accumulated Lift")
plt.legend(loc = 'upper right')
plt.show()



