function plot_test(data,label)

hold on;
if nargin ==1
    scatter(data(1,:), data(2,:), 'k');
else
    s = find(label == 0);
    scatter(data(1,s), data(2,s), 'b+');
    s = find(label == 1);
    scatter(data(1,s), data(2,s), 'r+');
end
hold off;
