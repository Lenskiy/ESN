classdef Reservoir < handle
    properties (SetAccess = private)
        x_updated;
    end
	properties
        nodes;
        numNodes;
        input_size;
        W;
        X_cur;
        X_init;
        parameters  = struct( 'neuron',     'tanh',...
                              'radius',      0, ...
                              'leakage',    0, ... 
                              'connectivity',0,...
                              'init_type', 'rand'); 
   end
   %~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   methods
      function obj = Reservoir(nodes, parameters)
          obj.parameters = parameters;
          obj.numNodes = length(nodes);
          obj.nodes = nodes;
          
          obj.X_cur = zeros(obj.numNodes, 1);
          obj.initialize(parameters.radius, parameters.connectivity);
          
          obj.x_updated = zeros(obj.numNodes, 1);
      end
      %~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
      %~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      function setInitStates(obj)
            obj.X_init = obj.X_cur;
      end
      %~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      function resetInitStates(obj)
            obj.X_cur = obj.X_init;
      end
      %~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      function output = forward(obj, input)
          for k = 1:obj.numNodes
            obj.x_updated(k) = obj.nodes(k).forward( input(k) + obj.W(k,:) * obj.X_cur );
          end
            obj.X_cur = (1 - obj.parameters.leakage) * obj.X_cur...
                        + obj.parameters.leakage * obj.x_updated;
            output = obj.X_cur;    
      end
      %~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      function error = backward(obj, x)
            error = obj.derFunction(x);
      end
   end
end