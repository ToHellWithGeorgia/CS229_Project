%%
features = 4:size(train_X,2);
X = [ones(size(train_X,1),1), train_X(:,features)]; % all image features
[B, FitInfo] = lasso(X, train_Y,'CV',5,'NumLambda',50);
lassoPlot(B,FitInfo,'PlotType','CV');

%%
X = [ones(size(train_X,1),1), train_X(:,features)]; 
yhat = X*B(:,FitInfo.IndexMinMSE) + FitInfo.Intercept(FitInfo.IndexMinMSE);
train_R2 = R2(train_Y, yhat)

%% validate
%X = validate_X;
%yhat = [ones(size(X,1),1), X(:,features)]*B(:,FitInfo.IndexMinMSE) ...
%    + FitInfo.Intercept(FitInfo.IndexMinMSE);
%figure; scatter(X(:,5), yhat); hold on;
%scatter(X(:,5), validate_Y,'r'); 
%validate_R2 = R2(validate_Y, yhat)

%% test
X = test_X;
yhat = [ones(size(X,1),1), X(:,features)]*B(:,FitInfo.IndexMinMSE) ...
    + FitInfo.Intercept(FitInfo.IndexMinMSE);
test_R2 = R2(test_Y, yhat)