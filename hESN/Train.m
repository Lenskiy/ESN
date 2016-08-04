classdef Train < handle
    properties
    end
   
    methods
        function obj = Train()
        end
        %~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        function [error] = train(obj, esn, input, target, initLen)
            X = zeros(sum(esn.architecture.numNodes) + esn.architecture.inputDim + 1, size(target,2));
            for j = 1:size(target, 2)
                u = input(:, j);
                X(1:sum(esn.architecture.numNodes), j) = esn.forward(u);
            end
            X(sum(esn.architecture.numNodes) + 1:end, :) = [ones(1, size(target,2)); input];
            Xinv = pseudoinverse(X(:, initLen + 1:end),[],'lsqr', 'tikhonov',...
               {@(x,r) r*normest(X)*x, 1e-4});
            esn.W_out =   target(:, initLen + 1:end) * Xinv;
            
%             X_ = X(:, initLen + 1:end);
%             Xinv =  X_' * inv(X_*X_' + 1e-4*eye(size(X_,1)));
%             esn.W_out =   target(:, initLen + 1:end) * Xinv;
            
            esn.setInitStates();
            Y = esn.generate(input, size(target,1), 1);
            error = mse(obj, target, Y);
            esn.resetInitStates();
      end
      %~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      function error = mse(obj, x, y)
            dif = x - y;
            error = sqrt(trace(dif' * dif));
      end
    end
end