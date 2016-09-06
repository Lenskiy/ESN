classdef BatchOutputLayerTrain < handle
    properties
    end
   
    methods
        function obj = BatchOutputLayerTrain()
        end
        %~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        function [error, x_collected] = train(obj, net, input, target, initLen)
            error = inf;
            outputId = net.getId('output');
            layersConnectedToOuput = net.getConnectedTo(outputId);
            indexesOfInterest = net.getInds(layersConnectedToOuput);

            x_interest = zeros(length(indexesOfInterest), size(target,2));
            for j = 1:size(target, 2)
                %j
                u = input(:, j);
                x_total = net.forward(u);
                x_interest(:, j) = x_total(indexesOfInterest);
            end
            x_collected = x_interest(:, initLen + 1:end);
          
            x_interest_inv =  x_collected' * inv(x_collected*x_collected' + 0.001*eye(size(x_collected,1)));
            W_out =   target(:, initLen + 1:end) * x_interest_inv;
            begInd = 1;
            for k = 1:length(layersConnectedToOuput)
                endInd = begInd + length(net.getInds(layersConnectedToOuput(k))) - 1;
                %begInd:endInd
                net.setWeights(outputId, layersConnectedToOuput(k),  W_out(:, begInd:endInd));
                begInd = endInd + 1;
            end
            
%             x_ = [x_collected(1:2,:); zeros(1, size(x_collected,2)); x_collected(3:end, :)];
%             y = net.W * x_;
%             figure, hold on; plot(y(3,:)'), plot(y(2,:)');
%             net.setInitStates();
%             Y = net.generate(input(:, 1), size(target,2), 1);
%             error = mse(obj, target, Y);
%             net.resetInitStates();
      end
      %~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      function error = mse(obj, x, y)
            dif = x - y;
            error = sqrt(trace(dif' * dif));
      end
    end
end