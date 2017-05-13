classdef trainClassifierOutputLayer < handle
    properties
    end
   
    methods
        function obj = trainClassifierOutputLayer()
        end
        %~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        function [error] = train(obj, net, input, target, initLen)
            
            %assert(size(input, 3) == size(target, 2), 'Number of samples in input and ouput should be equal')
            
            nExamples = size(input,2);
            
            outputId = net.getIdByType('output');
            toOuputIDs = net.getPrevNodes(outputId);
            
            for 
            net.setInput(input(:, k));
            
            net.rememberStates();
            
            

            Sinv =  input * inv(target'*target + 0.001*eye(size(target',1)));

            W_out =   Sinv*target';
            net.setWeightsForSelectedWeights(toOuputIDs, outputId, W_out');
           
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