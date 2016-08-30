%% Lorenz training/testing data
% [x y z] = lorenz(28, 10, 8/3);
% Lorenz_data = [x, y, z];

% mid_point = round((length(Lorenz_data) - 1) / 2);
% train_input = Lorenz_data(1:mid_point, :);
% train_output = Lorenz_data(2:mid_point + 1, :);
% test_input = Lorenz_data(mid_point + 1:end - 1, :);
% test_output = Lorenz_data(mid_point + 2:end, :);

%% Mackeyglass training/testing data
mackeyglass_data = mackeyglass(5000);
mid_point = round(19 * length(mackeyglass_data) / 20);
train_input = mackeyglass_data(1:mid_point, :);
train_output = mackeyglass_data(2:mid_point + 1, :);
test_input = mackeyglass_data(mid_point + 1:end - 1, :);
test_output = mackeyglass_data(mid_point + 2:end, :);


N = [10, 10];
connectivity = [0.1 0.1];
sp_radius = [0.01 2];
lr = [0.1 0.1];
sigma_noise = 0.005;

[Win, W] = buildStackedESN(size(train_input,2), N, connectivity, sp_radius);
[Wout, states, states_evolution] = trainStackedESN(train_input, train_output, Win, W, lr, sigma_noise);
Y = runStackedESN(test_input(1,:), length(test_output), states, Win, W, Wout, lr, sigma_noise);

figure, hold on
d1 = plot(states_evolution(1:N(1),:)','r');
d2 = plot(states_evolution(N(1)+1:end,:)','k');
title('States during the learning stage')
legend([d1(1), d2(1)],['\rho_1 = ', num2str(sp_radius(1))], ['\rho_2 = ', num2str(sp_radius(2))]);

figure(3), hold on;
title('Mackeyglass system');
plot(test_output(:,1));
plot(Y(:,1));

% figure(3), hold on;
% title('Lorenz system');
% plot3(test_output(:,1), test_output(:,2), test_output(:,3));
% plot3(Y(:,1), Y(:,2), Y(:,3));

%% Find the a best combination of spectral radii for ESN1 and ESN2 and the degree of connectivity
lr = [0.1 0.1];
sigma_noise = 0.005;
N = [10 10];
connectivity=0.05:0.05:0.5;
sp_radius1=0.05:0.1:1;
sp_radius2=0.05:0.1:1;
errorLen = length(test_input);
err_tensor_m = zeros(length(connectivity), length(sp_radius1), length(sp_radius2));
err_tensor_s = zeros(length(connectivity), length(sp_radius1), length(sp_radius2));
err_tensor_cor = zeros(length(connectivity), length(sp_radius1), length(sp_radius2));
for c = 1:length(connectivity)
    for r2 = 1:length(sp_radius2);
        for r1 = 1:length(sp_radius1);
            clear mse;
            for i = 1:100
                i
                [Win, W] = buildStackedESN(size(train_input,2), N, [connectivity(c) connectivity(c)], [sp_radius1(r1); sp_radius2(r2)]);
                [Wout, states, states_evolution] = trainStackedESN(train_input, train_output, Win, W, lr, sigma_noise);
                Y = runStackedESN(test_input(1,:), length(test_output), states, Win, W, Wout, lr, sigma_noise); 
                dif = Y(1:errorLen, :) - test_output(1:errorLen, :);
                mse(i) = sqrt(trace(dif' * dif));
            end
            
            err_tensor_m(c, r1, r2) = mean(mse(find(mse < 10000)));
            err_tensor_s(c, r1, r2) = std(mse(find(mse < 10000)));
            err_tensor_cor(c, r1, r2) = max(xcorr(test_output, Y, 'coeff'));
            disp(   ['c = ',num2str( connectivity(c), '%2.2f'),', r1 = ', num2str( sp_radius(r1), '%2.2f'),', r2 = ',num2str( sp_radius(r2), '%2.2f' ),', m = ',num2str( err_tensor_m(c, r1, r2) ),', s = ',num2str(err_tensor_s(c, r1, r2)),', c = ',num2str( err_tensor_cor(c, r1, r2) )] );
        end
    end
end