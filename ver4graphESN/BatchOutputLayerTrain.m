classdef BatchOutputLayerTrain < handle
    properties
    end
   
    methods
        function obj = BatchOutputLayerTrain()
        end
        %~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        function x_collected = train(obj, net, input, target, initLen)
            outputId = net.getIdByName('output');
            toOuputIDs = net.getPrevNodes(outputId);
            
           x_interest = zeros(net.getNumberOfStates(toOuputIDs), size(target,2));
           for j = 1:size(target, 2)
                u = input(:, j);
                net.forward(u);
                x_interest(:, j) = net.getStates(toOuputIDs);
           end
            
            x_collected = x_interest(:, initLen + 1:end);
            x_interest_inv = pseudoinverse(x_collected,[],'lsqr', 'tikhonov',...
               {@(x,r) r*normest(x_interest)*x, 1e-4});
%              x_interest_inv =  x_collected' * inv(x_collected*x_collected' + 0.001*eye(size(x_collected,1)));
            W_out =   target(:, initLen + 1:end) * x_interest_inv;
       
            % put the trained weights in the weight matrix of the network 
             net.setWeightsForSelectedWeights(outputId, toOuputIDs, W_out);
      end
      %~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      function error = mse(obj, x, y)
            dif = x - y;
            error = sqrt(trace(dif' * dif));
      end
    end
end