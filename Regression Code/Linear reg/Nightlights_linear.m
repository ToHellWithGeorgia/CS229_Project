select_features = 3;
X = [ones(size(train_X,1),1),train_X(:,select_features)]; % select your features here.
b = regress(train_Y, X);
yhat = X*b;

figure; scatter(train_X(:,1), yhat); hold on;
scatter(train_X(:,1), train_Y,'r'); 

train_R2 = 1 - sum((train_Y - yhat).^2)/sum((train_Y - mean(train_Y)).^2)
%% test
yhat_test = [ones(size(test_X,1),1), test_X(:,select_features)]*b;
figure; scatter(test_X(:,1), yhat_test); hold on;
scatter(test_X(:,1), test_Y,'r'); 
test_R2 = 1 - sum((test_Y - yhat_test).^2)/sum((test_Y - mean(test_Y)).^2)