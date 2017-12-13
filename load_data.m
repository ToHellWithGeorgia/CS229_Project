load('all_countries_dhs.mat');
all_Y=dhs(:,4);
survey_X=dhs(:,1:3);
featureX=dhs(:,5:4100);
all_X=[survey_X, featureX];
m = size(all_X,1);
ind = randperm(m);

train_X = all_X(ind(1:floor(m*0.67)),:);
train_Y = all_Y(ind(1:floor(m*0.67)),:);

% validate_X = all_X(ind(ceil(m*0.33*0.67):floor(m*0.33)),:);
% validate_Y = all_Y(ind(ceil(m*0.33*0.67):floor(m*0.33)),:);

test_X = all_X(ind(ceil(m*0.67):end),:);
test_Y = all_Y(ind(ceil(m*0.67):end),:);

R2 = @(Y, yhat) 1 - sum((Y - yhat).^2)/sum((Y - mean(Y)).^2);