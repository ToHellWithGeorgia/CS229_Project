# Using XGBoost to train models based on correlation features, feature
# forward search features, and lasso features.
# @author Zhihan Jiang

import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
import scipy.io as spio
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.grid_search import GridSearchCV

############################################################

mat= spio.loadmat('NGdhs.mat', squeeze_me=True)
NGdhs=mat['NGdhs']
mat1= spio.loadmat('all_countries_dhs.mat', squeeze_me=True)
dhs = mat1['dhs']
# Lasso feature trained for Nigeria
matlasso = spio.loadmat('feature_lasso.mat', squeeze_me=True)
featureidx_lasso_1 = matlasso['FeaturesLasso1SE']
featureidx_lasso_2 = matlasso['FeaturesLassoMinMSE']

# Lasso feature trained for all Africa
# matlasso_all = spio.loadmat('feature_lasso_all66.mat', squeeze_me=True)
# featureidx_lasso_all1 = matlasso_all['features_1SE']
# featureidx_lasso_all2 = matlasso_all['features_MinMSE']
matlasso_all = spio.loadmat('feature_lasso_ultimate.mat', squeeze_me=True)
featureidx_lasso_all1 = matlasso_all['feature_MSE']
featureidx_lasso_all2 = matlasso_all['feature_MSE']

# Highest correlation index
matcorr = spio.loadmat('features_correlation.mat', squeeze_me=True)
featureidx_correlation = matcorr['ind']

# Forward search index
matforward = spio.loadmat('Feature_forward_314.mat', squeeze_me=True)
featureidx_forward = matforward['F']

NGfeature_lasso_22 = NGdhs[:, featureidx_lasso_all1 + 2]
NGfeature_lasso_69 = NGdhs[:, featureidx_lasso_all2 + 2]
allfeature_lasso_22 = dhs[:, featureidx_lasso_all1 + 2]
allfeature_lasso_69 = dhs[:, featureidx_lasso_all2 + 2]
allfeature_correlation = dhs[:, featureidx_correlation]
allfeature_forward = dhs[:, featureidx_forward]

use_corr = False
use_lasso = False
use_nightlight = False
use_forward = True

# if (use_lasso):
# 	All_NGfeature = NGfeature_lasso_69
# elif (use_nightlight):
# 	All_NGfeature = NGdhs[:,[2,2]]
# 	# print All_NGfeature.shape
# else:
# 	# Nigeria data
# 	# Correlation higher than row 3 (need to subtract 1 as python index)
# 	# 1013 1522 762 1965 3821 3741 4060
# 	survey_X=NGdhs[:,0:2];
# 	# survey_X=NGdhs[:,2];
# 	featureX=NGdhs[:,4:4099];
# 	# Try smaller feature
# 	# featureX=NGdhs[:,2];
# 	All_NGfeature = np.concatenate((survey_X,featureX),axis=1)

# All_NGy=NGdhs[:,3]

# All data
if (use_corr):
	All_feature = allfeature_correlation
elif (use_lasso):
	All_feature = allfeature_lasso_69
elif (use_nightlight):
	All_feature = dhs[:,[2,2]]
elif(use_forward):
	All_feature = allfeature_forward
else:
	survey_X=dhs[:,4:5];
	# survey_X=dhs[:,2];
	featureX=dhs[:,5:204];
	# Try smaller feature
	# featureX=dhs[:,2];
	All_feature = np.concatenate((survey_X,featureX),axis=1)

All_y=dhs[:,3]


Seed = 2400
# x_NGtrain, x_NGtest, y_NGtrain, y_NGtest = \
# 	train_test_split(All_NGfeature, All_NGy,test_size=0.66, random_state=Seed)

x_train, x_test, y_train, y_test = \
	train_test_split(All_feature, All_y, test_size=0.33, random_state=Seed)

# max_depth=3, learning_rate=0.1, n_estimators=100, gamma=0, min_child_weight=1,
# max_delta_step=0, subsample=1, colsample_bytree=1, colsample_bylevel=1,
# reg_alpha=0, reg_lambda=1, scale_pos_weight=1, base_score=0.5, seed=None, missing=None, 

# print len(All_feature), len(All_y)
# Model that only uses Nigeria's training data
# model = xgb.XGBRegressor()
# model.fit(x_NGtrain, y_NGtrain)

# parameter testing
depth_range = range(2,3,1)
child_range = range(9,10,1)
estimator_range = range(7480,7500,20)
gamma_range = range(1,2,1)
alpha_range = range(1,2,1)
lambda_range = range(2,3,1)
subsample_range = [p/10.0 for p in range(5,6,1)]
colsample_range = [p/10.0 for p in range(5,6)]

eval_set = [(x_test, y_test), (x_train, y_train)]
fh = open('testresult2', 'w+')


scores = []
for depth in depth_range:
	for child in child_range:
		for estimator in estimator_range:
			for gamma in gamma_range:
				for alpha in alpha_range:
					for lmbda in lambda_range:
						for subsample in subsample_range:
							for colsample in colsample_range:

								model_all = xgb.XGBRegressor(
											max_depth=depth, 
											learning_rate=0.1, 
											n_estimators=estimator, 
											gamma=gamma,
											nthread=4, 
											min_child_weight=child,
											max_delta_step=0, 
											subsample=subsample, 
											colsample_bytree=colsample, 
											colsample_bylevel=1,
											reg_alpha=alpha, 
											reg_lambda=lmbda, 
											scale_pos_weight=1, 
											base_score=0.5)

								model_all.fit(x_train, y_train
										#eval_set = eval_set, eval_metric='rmse'
										)
								y_pred = model_all.predict(x_test)
								test_score = r2_score(y_pred, y_test)
								y_trainpred = model_all.predict(x_train)
								train_score = r2_score(y_trainpred, y_train)
								# scores.append(score)
								print 'depth:', depth, 'child:', child, \
										'estimator:', estimator, 'aplha:', alpha, \
										'lambda:', lmbda, 'subsample:', subsample, \
										'gamma:', gamma,\
										'train_score:', train_score, 'test_score:', test_score
								print >>fh, 'depth:', depth, 'child:', child, \
										'estimator:', estimator, 'aplha:', alpha, \
										'lambda:', lmbda, 'subsample:', subsample, \
										'gamma:', gamma,\
										'train_score:', train_score, 'test_score:', test_score


#print(model_all.booster().get_score(importance_type='weight'))

# print(model_all)

# param_test1 = {
# 	# 'max_depth':range(3,6,1),
# 	# 'min_child_weight':range(3,6,1)
# 	# 'n_estimators':range(50,1000,300)
# 	'reg_alpha':range(5,7,1),
# 	#'reg_lambda':range(0,3,1)
# }

# gsrch = GridSearchCV(estimator = xgb.XGBRegressor(
# 		max_depth=4, 
# 		learning_rate=0.1, 
# 		n_estimators=1000, 
# 		gamma=0, 
# 		min_child_weight=4,
# 		max_delta_step=0, 
# 		subsample=1, 
# 		colsample_bytree=1, 
# 		colsample_bylevel=1,
# 		# reg_alpha=0, 
# 		reg_lambda=1, 
# 		scale_pos_weight=1, 
# 		base_score=0.5
# 		),
# 		param_grid = param_test1, scoring = 'r2')
# gsrch.fit(x_train, y_train)

# print gsrch.grid_scores_ 
# print gsrch.best_params_ 
# print gsrch.best_score_
# print(model)

# # Use trained model of Nigeria to test Nigeria
# y_NGpred = model.predict(x_NGtest)
# NGloss = mean_squared_error(y_NGpred, y_NGtest)
# NG_r2 = r2_score(y_NGpred, y_NGtest)

# # Use trained model of Nigeria to test whole Africa
# y_allpred = model.predict(All_feature)
# allloss = mean_squared_error(y_allpred, All_y)
# all_r2 = r2_score(y_allpred, All_y)

# Use trained model of whole Africa to test whole Africa
# y_allpred_alltrain = model_all.predict(x_test)
# y_trainpred_alltrain = model_all.predict(x_train)
# # allloss_alltrain = mean_squared_error(y_allpred_alltrain, y_test)
# # trainloss_alltrain = mean_squared_error(y_trainpred_alltrain ,y_train)
# all_r2_alltrain = r2_score(y_allpred_alltrain, y_test)
# train_r2_alltrain = r2_score(y_trainpred_alltrain ,y_train)

# print 'NGtrain_NGMSEloss:',NGloss, 'NG_r2:', NG_r2
# print 'NGtrain_allMSEloss:', allloss, 'overall_r2:', all_r2
		#'overall_allMSEloss:', allloss_alltrain, \
		# 'trainset_MSEloss:', trainloss_alltrain, \
# print	'testset_r2_all:', all_r2_alltrain, \
# 		'trainset_r2', train_r2_alltrain, \
# 		'gsrch_test_r2', gsrch_r2, \
# 		'gsrch_train_r2', gsrch_train_r2
		

# Use linear regression
# regr = linear_model.LinearRegression()
# regr.fit(x_NGtrain, y_NGtrain)

# Make predictions using the testing set
# y_LR_NGpred = regr.predict(x_NGtest)
# LR_NGloss = mean_squared_error(y_LR_NGpred, y_NGtest)

# print 'Linear_NG_loss:', LR_NGloss




