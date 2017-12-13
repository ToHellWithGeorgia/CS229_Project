# Poverty Prediction by Selected Remote Sensing CNN Features

## Getting Started

For this project, we deliver a research on the poverty prediction using remote sensing CNN features. By carefully choosing features from the 4096 features provided by the CNN, we train a model that can predict wealth indices better than using the nightlights intensities. We conduct our research in 2 parts, feature selection and model training. We use correlation-based, Lasso-based and forward search methods to select features. And we use linear regression, ridge regression, Lasso regression and XGBoost to train our model and compare the performance. You can use the code we provide to go through this process.

### Prerequisites

* The feature selection methods and basic regression models are developed using built-in functions provided by MATLAB. 

* "all_countries_dhs.mat" is the file containing all the training data and training sets.

* To run our XGBoost code and VAE code in Python, you need:
  - Python 2.7
  - [XGBoost](http://xgboost.readthedocs.io/en/latest/build.html)
  - [scikit-learn](http://scikit-learn.org/stable/install.html)
  - [scipy kit](https://www.scipy.org/install.html)
  - [keras](https://keras.io/#installation)

### Installing

Please refer to the link above on how to install the dependencies.

For MacOS, if you have pip installed on your computer, you can do:

```
pip install xgboost
pip install -U scikit-learn
python -m pip install --user numpy scipy matplotlib ipython jupyter pandas sympy nose
pip install keras
```

You should be able to run our project code after installing the dependencies. Note you might need to install Tensorflow for Keras

## Running the tests

All the code in this project can be run easily without compiling.

### Feature selection and regression models (Matlab)

We have separated MATLAB files implementing all the feature selections and regression models. We will start by extracting useful CNN features from the provided dataset.

* Run "PCA_select.m", "correlation_feature_selection.m", "feature_forward_search.m" to get selected features with these three methods. In "correlation_feature_selection.m", it may be necessary to change the initial array length to get more selected features. For forward search, it takes a long time. Possible outcome is "feature_forward_search_314.mat", which contains 314 features (numbered from 4-4099). Features appear in the order they are selected. Run "All_lasso.m" to see features selected by Lasso regression. This takes a long time. Possible outcome is "Features_from_Super_Big_Lasso_67-33.mat", in which the selected features are indexed from 4-4099. (First 3 features are not used.) 

* After we obtained the selected features, we can use them to train different regression models:

  - Linear regression:
  Enter subdirectory "Linear reg". "All_linearReg.m" is running linear regression on all features, which will lead to overfitting. "Nightlights_linear.m" uses only nightlight feature, which is our baseline. "Select_linearReg.m" is basically the same as "All_linearReg.m", except that it requires the user overwrite the "select_features" variable (otherwise it cannot run).

  - Ridge regression:
  Enter subdirectory "ridge reg". This is similar to the linear regression directory, except that ridge regression is used instead of linear regression. The user is expected to overwrite "select_featuers" variable in "select_ridge.m".

  - Lasso regression:
  Enter subdirectory "Lasso reg". This is similar to the linear regression directory, except that lasso regression is used instead of linear regression. The user is expected to overwrite "select_featuers" variable in "select_lasso.m". The user should change arguments to the lasso() function to speed up.

* To generate intersection and union features:
  To do this part, the user is expected to run "correlation_feature_selection.m" first to generate a feature set of desired length, then use standard MATLAB functions to compute unions and intersections of the correlation-based feature set, the Lasso feature set and the forward search feature set. The latter two can be found in "feature_forward_search_314.mat" and "Features_from_Super_Big_Lasso_67-33.mat". As a convenience, the union and intersection used in our experiment is included in "intersection_union.mat". The user can directly put variables "union" and "in2" to the regression code to overwrite "select_features" as instructed above.

### XGBoost (Python)

For the XGBoost part, we have tuned the parameters for the best performance. And for the default setting, we use the features from the forward search method. You can see a list of parameters we choose and the train/test R2 score.

If you want to see results from other methods, you can simply change the boolean on line 48-51. We provide 4 different set of features to test.

```
python testxgboost.py
> depth: 2 child: 9 estimator: 7480 aplha: 1 lambda: 2 subsample: 0.5 gamma: 1 train_score: 0.874769893898 test_score: 0.509435929336
```

### VAE (Python)

You can simply run the command to see the result.

```
python VAEdhs.py
``` 

## Deployment

There might be version problems with Tensorflow, Keras and XGBoost. If you are having trouble compiling the code, you might need to use an older version of the library.

## Authors

* **Zhihan Jiang** - *XGBoost and documentation*
* **Yicheng Li** - *Feature Selection and Regression on MATLAB*
* **Zhaozhuo Xu** - *CNN Data and VAE*

## Acknowledgments

* We used some of the skeleton or demo code from the official site.
* This project is inspired by Jean, N.; Combining satellite imagery and machine learning to predict poverty. Science 353(6301):790–794.
* We love CS229

