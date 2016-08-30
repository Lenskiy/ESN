[x y z] = lorenz(28, 10, 8/3);
Lorenz_data = [x, y, z];

mid_point = round((length(Lorenz_data) - 1) / 2);
train_input = Lorenz_data(1:mid_point, :);
train_output = Lorenz_data(2:mid_point + 1, :);
test_input = Lorenz_data(mid_point + 1:end - 1, :);
test_output = Lorenz_data(mid_point + 2:end, :);

N = 100;
connectivity = 0.2;
sp_radius = 5;
lr = 0.1;
[Win, W] = buildESN(size(Lorenz_data,2), N, connectivity, sp_radius);
[Wout, states, states_evolution] = trainESN(train_input, train_output, Win, W, lr);
Y = runESN(test_input(1,:), length(test_output), states, Win, W, Wout, lr, .005);


errorLen = mid_point;
dif = Y(1:errorLen, :) - test_output(1:errorLen, :);
mse = sqrt(sum(diag(dif' * dif)));
disp( ['MSE = ', num2str( mse, '%5.2f' )] );


%[bc, bl, CorMat, LagMat] = esn_cor_best_match(test_output(1:errorLen, :), Y(1:errorLen, :));

figure(5);
subplot(2,1,1,'align');hold on;

time = [1:mid_point];
plot(time, test_output(1:mid_point,:), 'k');
%time_offset = bsxfun(@minus, ones(size(Y,2),1) * time, bl');
%time_offset = bsxfun(@minus, ones(size(Y,2),1) * time, bl * ones(size(Y,2),1));
time_offset = time;
plot(time_offset', Y(1:mid_point,:));
hold off;
axis tight;
legend('Target signal','Target signal', 'Target signal',  'Generated',  'Generated',  'Generated');
Y_ = Y;
% if (bl < 0)
%     Y_(1 : (mid_point + bl + 1), :) = Y(-bl:mid_point, :);
% else
%     Y_(bl + 1 : mid_point + bl ,:) = Y(1:mid_point, :);
% end

subplot(2,1,2);
hold on
time_sim = 1:3000;
plot3(test_input(time_sim,1), test_input(time_sim,2), test_input(time_sim,3),':.');
plot3(Y_(time_sim,1), Y_(time_sim,2), Y_(time_sim,3),':.');
for t = time_sim(1):20:time_sim(end)
    line([test_input(t,1) Y_(t,1)], [test_input(t,2) Y_(t,2)], [test_input(t,3) Y_(t,3)], 'color', 'k');
end

figure,
plot(abs(test_input(time_sim,:) - Y_(time_sim,:)))

    s1 = test_output(1:errorLen, 1);
    s2 = Y(1:errorLen, 1);
    [acor,lag] = xcorr(s1 , s2, 'coeff');
    [~,I] = max(abs(acor));
    timeDiff = lag(I)         % sensor 2 leads sensor 1 by 350 samples
    subplot(311); plot(s1); title('s1');
    subplot(312); plot(s2); title('s2');
    subplot(313); plot(lag,acor);
    title('Cross-correlation between s1 and s2')