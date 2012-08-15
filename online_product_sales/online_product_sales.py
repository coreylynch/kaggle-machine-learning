'''
Author: Corey Lynch
Date: 5/10/12
'''
from pandas import DataFrame
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import Scaler
from sklearn.cross_validation import train_test_split
from datetime import datetime,timedelta
from errorcurves import ErrorCurves
import numpy as np
from sklearn import mixture
import pandas

df = pandas.read_csv('TrainingDataset.csv')
df_test = pandas.read_csv('TestDataset.csv')
ids = df_test.pop('id')

outcomes = list()
train_sets = list()

quants = [i for i in df.columns if 'Q' in i]
df_quants = df[quants]
scaler = Scaler()
scaled = scaler.fit_transform(df_quants.fillna(0))
dpgmm = mixture.DPGMM(n_components = 75)
dpgmm.fit(scaled)
clusters = dpgmm.predict(scaled)
df['clusters'] = clusters

# Parse dates
jan1 = datetime(2000,1,1)

# Drop all rows where response variable == NaN
for i in range(1,13):
	df_i = df[df['Outcome_M'+str(i)]>0]
	outcomes.append(df_i.pop('Outcome_M'+str(i)))
	[df_i.pop(i) for i in df_i.columns if 'Out' in i]

	#drop nas first
	df_i['Date_1'] = df_i['Date_1'].fillna(0)
	df_i['Date_2'] = df_i['Date_2'].fillna(0)
	time_deltas_1 = [timedelta(int(i)) for i in df_i['Date_1'].values]
	time_deltas_2 = [timedelta(int(i)) for i in df_i['Date_2'].values]
	actual_dates_1 = [jan1+i for i in time_deltas_1]
	actual_dates_2 = [jan1+i for i in time_deltas_2]
	month_1 = [i.month for i in actual_dates_1]
	month_2 = [i.month for i in actual_dates_2]
	year_1 = [i.year for i in actual_dates_1]
	year_2 = [i.year for i in actual_dates_2]
	df_i['month_1'] = month_1
	df_i['month_2'] = month_2
	df_i['year_1'] = year_1
	df_i['year_2'] = year_2

	# Fillnas to zero
train_sets.append(df_i.fillna(0))

# Log response variables
for i in range(len(outcomes)):
	outcomes[i] = np.log(outcomes[i])

df_test_quants = df_test[quants]
scaled_test = scaler.transform(df_test_quants.fillna(0))
clusters_test = dpgmm.predict(scaled_test)
df_test['clusters'] = clusters_test

df_test = df_test.fillna(0)
time_deltas_1_test = [timedelta(int(i)) for i in df_test['Date_1'].values]
time_deltas_2_test = [timedelta(int(i)) for i in df_test['Date_2'].values]
actual_dates_1_test = [jan1+i for i in time_deltas_1_test]
actual_dates_2_test = [jan1+i for i in time_deltas_2_test]
month_1_test = [i.month for i in actual_dates_1_test]
month_2_test = [i.month for i in actual_dates_2_test]
year_1_test = [i.year for i in actual_dates_1_test]
year_2_test = [i.year for i in actual_dates_2_test]
df_test['month_1'] = month_1_test
df_test['month_2'] = month_2_test
df_test['year_1'] = year_1_test
df_test['year_2'] = year_2_test


assert len(train_sets[0].columns)==len(df_test.columns)

x_train_sets= list()
x_test_sets = list()
y_train_sets = list()
y_test_sets = list()

# to be transformed by RFs
transformed_trains = list()
tf_x_train_sets= list()
tf_x_test_sets = list()
tf_y_train_sets = list()
tf_y_test_sets = list()


for i in range(len(outcomes)):
	x_train, x_test, y_train, y_test = train_test_split(train_sets[i],outcomes[i],test_size=0)
	x_train_sets.append(x_train)
	x_test_sets.append(x_test)
	y_train_sets.append(y_train)
	y_test_sets.append(y_test)


# Train a separate random forest for each time period
rf_models = list()

n_trees = 2000
for i in range(1,13):
	rf_models.append(RandomForestRegressor(n_estimators=n_trees,bootstrap=True, compute_importances=True, oob_score=True,n_jobs=-1))

#########################################################################
# Train for model estimation 
for i in range(len(rf_models)):
	print 'training '+str(i)
	rf_models[i].fit(x_train_sets[i],y_train_sets[i])
	
def rmsle(preds,actuals):
	return np.sum((np.log(np.array(preds)+1) - np.log(np.array(actuals)+1))**2 / len(preds))

# test the models
def test(list_of_models):
	preds = np.array(list())
	actuals = np.array(list())
	for i in range(len(list_of_models)):
		preds = np.concatenate((preds,np.exp(list_of_models[i].predict(x_test_sets[i]))))
		actuals = np.concatenate((actuals,y_test_sets[i]))
		# need to exponentiate actuals, since we logged them earlier
	return rmsle(preds,np.exp(actuals))

def test_bin(list_of_models):
	preds = np.array(list())
	actuals = np.array(list())
	for i in range(len(list_of_models)):
		preds = np.concatenate((preds,np.exp(list_of_models[i].predict(x_test_sets[i]))))
		# bin the preds
		preds = np.array([bin_pred(j) for j in preds])
		actuals = np.concatenate((actuals,y_test_sets[i]))
		# need to exponentiate actuals, since we logged them earlier
	return rmsle(preds,np.exp(actuals))


def oob_estimate(list_of_models):
	oob_list = list()
	for i in list_of_models:
		oob_list.append(i.oob_score_)
	return np.mean(np.array(oob_list))

def bin_pred(i):
	return (round(i,-3) if i>1250 else 500.00)


# Save models
with open('/models/rf_%s' % str(n_trees), 'wb') as f:
	pickle.dump(rf_models, f)

# Show variable importances
variable_importances = [sorted(zip(i.feature_importances_,train_sets[0].columns),reverse=True) for i in rf_models]
for j in range(len(variable_importances)):
    plt.pie(variable_importances[j],labels=train_sets[0].columns)
    title = 'Variable Importances for RF 2000, Month %s' % str(j+1)
    plt.title(title)
    plt.savefig('pie_rf_2000_m%s.png' % str(j+1))
    plt.clf()

#  Plot error curves
model_name = 'RF_%s' % str(n_trees)
for i in range(len(rf_models)):
	ecurves = ErrorCurves(rf_models[i],train_sets[i].values,outcomes[i].values,model_name+'_M'+str(i+1))
	ecurves.plot_and_save()


#########################################################################

# Train on the full dataset for a kaggle submission 
for i in range(1,13):
	rf_models.append(RandomForestRegressor(n_estimators=200,n_jobs=-1))

for i in range(len(rf_models)):
	print 'training '+str(i)
	rf_models[i].fit(x_train_sets[i],y_train_sets[i])

# Prepare for submission to kaggle
kaggle_preds = []
for i in range(len(rf_models)):
	kaggle_preds.append(np.exp(rf_models[i].predict(df_test)))


df_submit = DataFrame(kaggle_preds).transpose()
df_submit.to_csv('kaggle_submit.csv',header=False,index=False)


