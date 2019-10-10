function plot_train(data,label)

hold on;
s = find(label == 0);
scatter(data(1,s), data(2,s), 'bo', 'filled');
s = find(label == 1);
scatter(data(1,s), data(2,s), 'ro', 'filled');
hold off;