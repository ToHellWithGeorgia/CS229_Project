% Correlation-based feature selection
% figure 1. check correlation of features with asset_index (Y)
X = train_X;
X(:,1:3) = ones(size(X,1),3);
C1 = corr(X, train_Y);
val = zeros(600,1);
ind = zeros(600,1);
for count = 1:600
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

X = train_X(:, select_features);
b = ridge(train_Y, X, 1, 0); % ridge parameter
X = [ones(size(X,1),1), X];

yhatr = X*b;
% figure; scatter(X(:,2), yhatr); hold on;
% scatter(X(:,2), train_Y,'r'); 

train_R2 = 1 - sum((train_Y - yhatr).^2)/sum((train_Y - mean(train_Y)).^2)

%%
X = test_X(:,select_features);
yhat_test = [ones(size(test_X,1),1), X]*b;
% figure; scatter(test_X(:,1), yhat_test); hold on;
% scatter(test_X(:,1), test_Y,'r'); 
test_R2 = 1 - sum((test_Y - yhat_test).^2)/sum((test_Y - mean(test_Y)).^2)


