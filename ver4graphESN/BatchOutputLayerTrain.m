classdef BatchOutputLayerTrain < handle
    properties
    end
   
    methods
        function obj = BatchOutputLayerTrain()
        end
        %~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        function x_collected = train(obj, net, input, target, initLen, type)
            outputId = net.getIdByType('output');
            toOuputIDs = net.getPrevNodes(outputId);
            
           nExamples = size(target,2); 
           x_interest = zeros(net.getNumberOfStates(toOuputIDs), nExamples);
           for j = 1:nExamples
%                 if(mod(j, 1000) == 0)
%                     j/nExamples
%                 end
                u = input(:, j);
                net.forward(u);
                x_interest(:, j) = net.getStates(toOuputIDs);
           end
            
            x_collected = x_interest(:, initLen + 1:end);
            
            switch type
                case 'ridge1'
                    lambda = 1e-5;
                    x_interest_inv = pseudoinverse(x_collected,[],'lsqr', 'tikhonov',...
                      {@(x,r) r*normest(x_interest)*x, lambda});
                    W_out =   target(:, initLen + 1:end) * x_interest_inv;
                case 'ridge2'
                    lambda = 1e-5;
                     x_interest_inv =  x_collected' * inv(x_collected * x_collected' + lambda*eye(size(x_collected,1)));
                     W_out =   target(:, initLen + 1:end) * x_interest_inv;
                    
                    %W_out = ((x_collected * x_collected' + lambda*eye(size(x_collected,1))) \ x_collected * target(:, initLen + 1:end)')';
                case 'gd'
                    lambda = 1e-3;
                    W_out = obj.gradientDescent(x_collected, target(:, initLen + 1:end), lambda);
                case 'sgd'
                    lambda = 1e-5;
                    W_out = obj.stochasticGradientDescent(x_collected, target(:, initLen + 1:end), struct('lambda', lambda, 'batch', 0.1, 'tries', 10));                    
                case 'lasso'
                    lambda = 1e-3;
                    for k = 1:size(target,1)
                        k
                        [B, fitsinfo] = lasso(x_collected', target(k, initLen + 1:end));
                        W_out(k, :) = B(:,1);
                    end
            end
            err = W_out * x_collected - target(:, initLen + 1:end) + (lambda * sum(W_out.^2,2));
            mse = sum(sum(err.^2, 2))/(nExamples - initLen);
            % put the trained weights in the weight matrix of the network 
             net.setWeightsForSelectedWeights(toOuputIDs, outputId, W_out);
      end
      %~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      function error = mse(obj, x, y)
            dif = x - y;
            error = sqrt(trace(dif' * dif));
      end
      %~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      function [W, mse] = gradientDescent(obj, x, y, lambda)
          numIter = 1000;
          alpha = 1;
          W =  randn(size(y, 1), size(x, 1));
          trainingSize = size(x, 2);
          mse = zeros(numIter, size(y,1));
          % Find alpha for which the error on the first interation is smaller 
          err = (W * x - y) + (lambda * sum(W.^2,2));
          while (sum(sum(((W - alpha * err * x' / trainingSize)*x - y).^2, 2)) > sum(sum(err.^2, 2)))
            alpha = alpha / 2;
          end
          % Search for minimum
          for k = 1:numIter
            if(mod(k, 1000) == 0)
                disp (num2str(k/1000))
            end
            err = (W * x - y) + (lambda * sum(W.^2,2));
            mse(k, :) = sum(err.^2,2)/trainingSize;
            W = W - alpha * err * x' / trainingSize;
          end
          figure, plot(mse); title('Mean Square Error'); xlabel('iterations');ylabel('MSE');
      end
      %~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      function [W, mse] = stochasticGradientDescent(obj, X, Y, params)
          if nargin < 4 % default parameters
              params = struct('lambda', 0.001, 'batch', 0.1, 'tries', 10);
          end
          alpha = 1;
          trSize = size(X, 2);      % the size of the training set
          numOutputs = size(Y,1);
          batchSize = params.batch *  trSize;
          numBatches = ceil(trSize / batchSize);
          mse = zeros(numBatches * params.tries, numOutputs);
          
          % Set initial search position randomly
          W =  randn(size(Y, 1), size(X, 1));
          % Find alpha for which the error on the first GD step is smaller 
          err = (W * X - Y) + (params.lambda * sum(W.^2,2));
          mseInit = sum(sum(err.^2, 2));
          while (sum(sum(((W - alpha * err * X' / trSize)*X - Y).^2, 2)) > mseInit)
            alpha = alpha / 2; % decrease alpha a bit and recalculate error
          end
          alpha = alpha / 2; % even smaller alpha is a good start
          % Search for the minimum
          for k = 1:params.tries % 
            for l = 1:numBatches
            	if(mod(l, 10) == 0)
                    disp (['Completed: ', num2str(((k - 1) * numBatches + l)...
                                            /(params.tries*numBatches))]);
                end
                % Indexes that correspond the batch
                inds = ((l-1)*batchSize + 1):min(l*batchSize, trSize);
                batch = X(:,inds);
                err = (W * batch  - Y(inds) + (params.lambda * sum(W.^2,2)));
                % mse is used just for visualization of the error evolution
                mse((k - 1) * numBatches + l, :) = sum(err.^2,2)/batchSize; 
                W = W - alpha/k * err * batch' / batchSize;
            end
            %change the order of training vectors
            inds = randperm(trSize);
            X = X(:, inds);
            Y = Y(:, inds);
          end
         figure, plot(mse); title('Mean Square Error');
         xlabel('iterations');ylabel('MSE');
      end      
    end
end