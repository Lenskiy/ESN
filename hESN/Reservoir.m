classdef Reservoir < handle
	properties
        neuron;
        numNodes;
        leakage;
        input_size;
        W;
        X_cur;
        connectivity;
        radius;
   end
   %~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   methods
      function obj = Reservoir(type, numNodes, radius, leakage, connectivity)
          obj.numNodes = numNodes;
          obj.leakage = leakage;
          obj.radius = radius;
          obj.connectivity = connectivity;
          obj.X_cur = zeros(numNodes, 1);
          obj.initialize(radius, connectivity);
          obj.neuron = Neuron(type);
      end
      
      function initialize(obj, radius, connectivity)
            rng('shuffle');

            % generate connecting weight in reservoir
            obj.W = sprand(obj.numNodes, obj.numNodes, connectivity);
            obj.W(obj.W ~= 0) = obj.W(obj.W ~= 0) - 0.5;

            % compute spectral radius i.e. the largest absolute eigen value 
            opts.tol = 1e-3;
            maxVal = max(abs(eigs(obj.W, 1, 'lm', opts)));
            obj.W = radius * obj.W/maxVal; % normalize W
      end
      
      function output = forward(obj, input)
            x_updated = obj.neuron.forward( input + obj.W * obj.X_cur );
            obj.X_cur = (1 - obj.leakage) * obj.X_cur...
                        + obj.leakage * x_updated;
            output = obj.X_cur;    
      end
      
      function error = backward(obj, x)
            error = obj.derFunction(x);
      end
   end
end