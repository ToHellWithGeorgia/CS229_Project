select_features = 4:403;
X = train_X(:, select_features);
b = ridge(train_Y, X, 1, 0); % ridge parameter
X = [ones(size(X,1),1), X];

yhatr = X*b;
figure; scatter(X(:,2), yhatr); hold on;
scatter(X(:,2), train_Y,'r'); 

train_R2 = 1 - sum((train_Y - yhatr).^2)/sum((train_Y - mean(train_Y)).^2)

%% test
X = test_X(:,select_features);
yhat_test = [ones(size(test_X,1),1), X]*b;
figure; scatter(test_X(:,1), yhat_test); hold on;
scatter(test_X(:,1), test_Y,'r'); 
test_R2 = 1 - sum((test_Y - yhat_test).^2)/sum((test_Y - mean(test_Y)).^2)