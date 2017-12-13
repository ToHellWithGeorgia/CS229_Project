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
%%
F = [];
%%
count = length(F);
while (count <= 500)
    count = count + 1;
    test_R2 = zeros(4099,1);
   	for new_feature = 4:4099
        if (sum(F == new_feature) ~= 0)
            continue;
        end
        X = [train_X(:,[F,new_feature])]; % select your features here.
        b = ridge(train_Y, X, 1, 0); 
        yhat_test = [ones(size(test_X,1),1), test_X(:,[F,new_feature])]*b;
        test_R2(new_feature) = 1 - sum((test_Y - yhat_test).^2)/sum((test_Y - mean(test_Y)).^2);
    end
    [val(count),ind] = max(test_R2);
    F = [F, ind];
end

%% generate plot
for i = 1:300
    X = train_X(:, F(1:i));
    b = ridge(train_Y, X, 1, 0); 
    yhat_test = [ones(size(test_X,1),1), test_X(:, F(1:i))]*b;
    val(i) = 1 - sum((test_Y - yhat_test).^2)/sum((test_Y - mean(test_Y)).^2);
end

figure; plot(1:300, val(1:300));
xlabel('number of forward search features');
ylabel('test R2 score');
title('test R2 in forward search feature selection');