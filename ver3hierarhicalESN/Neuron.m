classdef Neuron < handle
   properties (SetAccess = private)
      actFunction = @tanh;
      derFunction = @(x) 1 - tanh(x).^2
   end
   
   methods
      function obj = Neuron(type)
         switch type
             case 'tanh'
                obj.actFunction =  @(x) tanh(x);
                obj.derFunction =  @(x) 1 - tanh(x).^2;
             case 'linear'
                obj.actFunction =  @(x) x;
                obj.derFunction = 1;
             case 'logistic'
                obj.actFunction =  @(x) 1 / (1 + exp(-x));
                obj.derFunction = @(x) exp(x) / (1 + exp(x)).^2;               
         end
         
         %obj(numNeurons) = Neuron(type, state);
      end 
      %~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      function X = forward(obj, X)
         X = obj.actFunction(X);
      end
      %~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      function X = backward(obj, X)
        X = obj.derFunction(X);
      end
   end
   
end