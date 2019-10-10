function dist = L2_distance(data_train, data_test)
% return a Ntrain * Ntest matrix of distances

ntrain = size(data_train, 2);
ntest = size(data_test, 2);

dist = zeros(ntrain, ntest);

for i = 1 : ntest
    for j = 1 : ntrain
        dist(j, i) = sum((data_test(:,i) - data_train(:,j)).^2);
    end
end

return
end
