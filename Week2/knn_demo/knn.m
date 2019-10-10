function [label_test] = knn(k, data_train, label_train, data_test)

error(nargchk(4,4,nargin));

dist = L2_distance(data_train, data_test);
[sorted_dist, nearest] = sort(dist);
nearest = nearest(1:k,:);
label_test = label_train(nearest);

% note this only works for binary labels
label_test = mean(label_test,1) >= 0.5;
