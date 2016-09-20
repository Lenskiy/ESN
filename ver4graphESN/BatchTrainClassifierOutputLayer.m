classdef BatchTrainClassifierOutputLayer < handle
    properties
    end
   
    methods
        function obj = BatchTrainClassifierOutputLayer()
        end
        %~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        function [error] = train(obj, net, input, target, initLen)
            
            nExamples = size(input,3);
            nNetworkInputs = size(input,2);
            nSamplesInExample = size(input,1);
            nClasses = size(target, 1);
            
            outputId = net.getIdByName('output');
            toOuputIDs = net.getPrevNodes(outputId);
                
            avgStates = zeros(net.getNumberOfStates(toOuputIDs), nExamples);
            
            net.rememberStates();
            
            for k = 1:nExamples
                k/nExamples
                net.recallStates();
               
                interestStates = zeros(net.getNumberOfStates(toOuputIDs), nSamplesInExample);
                for j = 1:nSamplesInExample
                    net.forward(input(:, j, k));
                    interestStates(:, j) = net.getStates(toOuputIDs);
                end          
                
                % Spatial average            
                avgStates(:, k) = mean(interestStates(:, initLen + 1:end),2);
            end
            
            
            Sinv =  avgStates' * inv(avgStates*avgStates' + 0.001*eye(size(avgStates,1)));

            W_out =   target * Sinv;
            net.setWeightsForSelectedWeights(outputId, toOuputIDs, W_out);
           
            % Calc classification errors
          
            P = W_out*avgStates;
            h=0;
            
            indTarget = vec2ind(target);
            for i = 1:nExamples
                [~, maxind] = max(P(:,i));  
                
                if maxind == indTarget(i)
                    h = h +1;
                end
            end
         
            % Hit rate
            error = 1 - h/nExamples;
            
        end
      %~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    end
end