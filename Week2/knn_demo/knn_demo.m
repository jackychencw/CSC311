clc;clear;clf;
% knn_demo

X_MAX = 50;
Y_MAX = 50;
NUM_TRAIN = 130;
RANDOM_LABELS = 0;

%-------------------------------------------------------------
% Generate the test points, on a regular grid.
%-------------------------------------------------------------
[test_x, test_y] = meshgrid(1:X_MAX, 1:Y_MAX);
data_test = [test_x(:) test_y(:)]';
% scatter(data_test(1,:), data_test(2,:), 'k.');

%-------------------------------------------------------------
% Generate random training points.
%-------------------------------------------------------------
data_train = zeros(2,NUM_TRAIN);
data_train(1,:) = rand(1,NUM_TRAIN)*X_MAX;
data_train(2,:) = rand(1,NUM_TRAIN)*Y_MAX;
if RANDOM_LABELS
    label_train = (rand(1,NUM_TRAIN) >= 0.5);
else
    label_train = data_train(2,:) >= ((data_train(1,:) - X_MAX/2).^2 /4 + 3);
    % add label noise
    label_train = xor(label_train, (rand(size(label_train)) >= 0.8));
end

plot_train(data_train, label_train);

%-------------------------------------------------------------
% Run kNN for various values of k, plotting the results
%-------------------------------------------------------------
set(gcf,'doublebuffer', 'on');
for k = [1 3 5 7 9]
    label_test = knn(k, data_train, label_train, data_test);
    pause;
    clf;
    title(sprintf('%d Nearest Neighbours', k));
    plot_test(data_test, label_test);
    plot_train(data_train, label_train);
    hold on;
    contour(reshape(label_test,[X_MAX Y_MAX]), 1, 'LineWidth', 3);
    hold off;
end

close all;
clear all;
