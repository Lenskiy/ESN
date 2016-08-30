%% Problem 2: Lorenz system
% https://www.math.auckland.ac.nz/~hinke/preprints/okh_icdea_preprint.pdf
% (a) Run the code in the following section;
% (b) Select the parameters of connectivity, speactral radius (sp_radius), and 
% leakage rate(lr) that generate a trajactory that follows the trajectory
% of the Lorenz attractor. Try to keep N lower, however the accuracy is the
% most important criterion; For every set of parameters run the ESN 10
% times, calculate the mean and standard deviation. Print out the set of 
% parameters that produces minimal MSE, and visualize the trajectory.
% (c) Explain based on your experience how each of the parameters effect
% the accuracy.
% [x y z] = lorenz(28, 10, 8/3, [0 1 1.05], [0 50], 0.000001);
% Lorenz_data = [x, y, z];
% 
mid_point = round(length(Lorenz_data) / 2);
train_input = Lorenz_data(1:mid_point, :);
train_output = Lorenz_data(2:mid_point + 1, :);
test_input = Lorenz_data(mid_point + 1:end - 1, :);
test_output = Lorenz_data(mid_point + 2:end, :);


N = 10;
connectivity=0.05:0.05:0.95;
sp_radius=0.5:0.1:3;
lr = 0.05:0.05:0.95;
errorLen = 1000;
err_tensor_m = zeros(length(connectivity), length(sp_radius), length(lr));
err_tensor_s = zeros(length(connectivity), length(sp_radius), length(lr));
for c = 1:length(connectivity)
    for r = 1:length(sp_radius);
        for l = 1:length(lr)
            clear mse;
            for i = 1:100
                %i
                [Win, W] = buildESN(size(Lorenz_data,2), N, connectivity(c), sp_radius(r));
                [Wout, states] = trainESN(train_input, train_output, Win, W, lr(l));
                Y = runESN(test_input(1,:), length(test_output), states, Win, W, Wout, lr(l)); 
                dif = Y(1:errorLen, :) - test_output(1:errorLen, :);
                mse(i) = sqrt(trace(dif' * dif));
            end
            
            err_tensor_m(c, r, l) = mean(mse(find(mse < 10000)));
            err_tensor_s(c, r, l) = std(mse(find(mse < 10000)));
            disp(   ['c = ',num2str( connectivity(c), '%2.2f'),', r = ', num2str( sp_radius(r), '%2.2f'),', l = ',num2str( lr(l), '%2.2f' ),', m = ',num2str( err_tensor_m(c, r, l) ),', s = ',num2str(err_tensor_s(c, r, l))] );
        end
    end
end