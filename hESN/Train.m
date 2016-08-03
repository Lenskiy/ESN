classdef Train < handle
    properties
    end
   
    methods
        function obj = Train()
        end
        %~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        function [error] = train(obj, esn, input, target, initLen)
            X = zeros(size(target,1), esn.architecture.numNodes + esn.architecture.inputDim + 1);
            for j = 1:size(target,1)
                u = input(j, :)';
                X(j, 1:esn.architecture.numNodes) = esn.reservoir.forward(esn.W_in * [1; u]);
            end
            X(:, esn.architecture.numNodes + 1:end) = [ones(size(target,1),1) input];
            Xinv = pseudoinverse(X(initLen + 1:end,:)',[],'lsqr', 'tikhonov',...
                {@(x,r) r*normest(X)*x, 1e-4});
            
            esn.W_out =   target(initLen + 1:end, :)' * Xinv;
            
            esn.reservoir.setInitStates();
            Y = esn.generate(input, size(target,1), 1);
            error = mse(obj, target, Y);
            esn.reservoir.resetInitStates();
      end
      %~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      function error = mse(obj, x, y)
            dif = x - y;
            error = sqrt(trace(dif' * dif));
      end
    end
end