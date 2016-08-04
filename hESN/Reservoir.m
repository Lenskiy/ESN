classdef Reservoir < handle
%     properties (SetAccess = private)
%         x_updated;
%     end
	properties
        unit;
        numNodes;
        input_size;
        W;
        X_cur;
        X_init;
        parameters  = struct( 'node_type',     'tanh',...
                              'radius',      0, ...
                              'leakage',    0, ... 
                              'connectivity',0,...
                              'init_type', 'rand'); 
   end
   %~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   methods
      function obj = Reservoir(numNodes, parameters)
          if nargin ~= 0
              obj(length(numNodes)) = Reservoir;
              for k = 1:length(numNodes)
                  obj(k).numNodes = numNodes(k);
                  obj(k).parameters = parameters(k);
                  obj(k).unit = Neuron(parameters(k).node_type);  
                  obj(k).X_cur = zeros(numNodes(k), 1);
                  obj(k).initW(parameters(1).radius, parameters(k).connectivity);
              end
%           obj.x_updated = zeros(obj.numNodes, 1);
          end
      end
      %~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      function initW(obj, radius, connectivity)
            %rng('shuffle');

            % generate connecting weight in reservoir
            obj.W = sprand(obj.numNodes, obj.numNodes, connectivity);
            obj.W(obj.W ~= 0) = obj.W(obj.W ~= 0) - 0.5;

            % compute spectral radius i.e. the largest absolute eigen value 
            opts.tol = 1e-3;
            maxVal = max(abs(eigs(obj.W, 1, 'lm', opts)));
            obj.W = radius * obj.W/maxVal; % normalize W
      end
      %~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      function setStates(obj, X)
            obj.X_cur = X;
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
%           for k = 1:obj.numNodes
%             obj.x_updated(k) = obj.nodes(k).forward( input(k) + obj.W(k,:) * obj.X_cur );
%           end
            x_updated = obj.unit.forward( input + obj.W * obj.X_cur );
            obj.X_cur = (1 - obj.parameters.leakage) * obj.X_cur...
                        + obj.parameters.leakage * x_updated;
            output = obj.X_cur;    
      end
      %~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      function error = backward(obj, x)
            error = obj.derFunction(x);
      end
   end
end