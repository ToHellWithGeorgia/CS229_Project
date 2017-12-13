% Correlation-based feature selection
X = train_X;
X(:,1:3) = ones(size(X,1),3);
C1 = corr(X, train_Y);
val = zeros(10,1);
ind = zeros(10,1);
for count = 1:10
   [val(count), ind(count)] = max(C1);
   C1(ind(count)) = -inf;
end

figure(1); 
bar(val);
set(gca,'xticklabel',ind);
xlabel('feature number in train X');
ylabel('correlation with asset_index');
ylim([0.65, 0.75]);

select_features = ind;


