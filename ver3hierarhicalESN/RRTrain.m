classdef RRTrain < handle
    properties
    end
   
    methods
        function obj = RRTrain()
        end
        %~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        function [error, X_collected] = train(obj, esn, input, target, initLen)
            X = zeros(sum(esn.architecture.numNodes) + esn.architecture.inputDim + 1, size(target,2));
            for j = 1:size(target, 2)
                u = input(:, j);
                X(1:sum(esn.architecture.numNodes), j) = esn.forward(u);
            end
            X_collected = X(1:sum(esn.architecture.numNodes), initLen:end);
            X(sum(esn.architecture.numNodes) + 1:end, :) = [ones(1, size(target,2)); input];
          
            
            X_ = X(:, initLen + 1:end);
            Xinv_ =  X_' * inv(X_*X_' + 0.001*eye(size(X_,1)));
            esn.W_out =   target(:, initLen + 1:end) * Xinv_;
            
            esn.setInitStates();
            Y = esn.generate(input(:, 1), size(target,2), 1);
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