% Correlation-based feature selection
% figure 1. check correlation of features with asset_index (Y)
X = train_X;
X(:,1:3) = ones(size(X,1),3);
C1 = corr(X, train_Y);
val = zeros(1000,1);
ind = zeros(1000,1);
for count = 1:1000
   [val(count), ind(count)] = max(C1);
   C1(ind(count)) = -inf;
end

figure(1); 
bar(val);
set(gca,'xticklabel',ind);
xlabel('feature number in train X');
ylabel('correlation with asset_index');
ylim([0.65, 0.75]);

%%
select_features = in2;

X = [ones(size(train_X,1),1), train_X(:,select_features)]; % all image features
[B, FitInfo] = lasso(X, train_Y,'CV',3,'NumLambda',20);
lassoPlot(B,FitInfo,'PlotType','CV');
%%
X = [ones(size(train_X,1),1), train_X(:,select_features)]; 
yhat = X*B(:,FitInfo.IndexMinMSE) + FitInfo.Intercept(FitInfo.IndexMinMSE);
train_R2 = R2(train_Y, yhat)

%% validate
X = validate_X;
yhat = [ones(size(X,1),1), X(:,select_features)]*B(:,FitInfo.IndexMinMSE) ...
    + FitInfo.Intercept(FitInfo.IndexMinMSE);
% figure; scatter(X(:,5), yhat); hold on;
% scatter(X(:,5), validate_Y,'r'); 
validate_R2 = R2(validate_Y, yhat)

%% test
X = test_X;
yhat = [ones(size(X,1),1), X(:,select_features)]*B(:,FitInfo.IndexMinMSE) ...
    + FitInfo.Intercept(FitInfo.IndexMinMSE);
test_R2 = R2(test_Y, yhat)
