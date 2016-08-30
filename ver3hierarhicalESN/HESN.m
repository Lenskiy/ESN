classdef HESN < handle
    properties (SetAccess = private)
        X_last;
        totalStates;
    end
    properties
        
        architecture;% = struct('inputDim',   0, ...
                     %         'numNodes',   [0], ... 
                     %         'outputDim',  0);      
        parameters;%  = struct( 'node_type',     'tanh',...
                   %           'radius',      0, ...
                   %           'leakage',    0, ... 
                   %           'connectivity',0,...
                   %           'init_type', 'rand');
                   
        hParameters% = struct('radius',      0,...
                   %         'leakage',      0,...
                   %         'connectivity', 0,...
                   %         'init_type', 'rand');
        reservoirs;
        
        W_in;      % Input weight matrix for all reservoirs
        W_fb;      % A cell array of fee
        W_out;
        W
        
        Y_last;
    end
   
    methods
      function obj = HESN(sArchitecture, hParameters, sParameters)
            obj.architecture = sArchitecture;
            obj.parameters = sParameters;
            obj.hParameters = hParameters;
            obj.reservoirs = Reservoir(sArchitecture.numNodes, sParameters);

            obj.totalStates = sum(sArchitecture.numNodes);
            obj.X_last = zeros(obj.totalStates,1);
            obj.initialize();
      end
      %~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      function initialize(obj)
            %rng('shuffle');
            initWin(obj, obj.hParameters.init_type);
            initWfb(obj, obj.hParameters.init_type);
            initW(obj, obj.hParameters.init_type, obj.architecture.topology);
            obj.Y_last = 0;
      end
      %~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      function initWin(obj, init_type)
            %initialize input weight matrices of each reservior
            for k = 1:length(obj.architecture.numNodes)
                switch (init_type)
                    case 'const'
                        obj.W_in{k} = ones(obj.architecture.numNodes(k),...
                            1 + obj.architecture.inputDim) / obj.architecture.numNodes;
                    case 'rand'
                        obj.W_in{k} = (rand(obj.architecture.numNodes(k),...
                            1 + obj.architecture.inputDim) - 0.5);
                        %obj.W_fb = (rand(obj.architecture.numNodes,...
                        %    obj.architecture.outputDim) - 0.5);
                    case 'randn'
                        obj.W_in{k} = (randn(obj.architecture.numNodes(k),...
                            1 + obj.architecture.inputDim));
                end
            end      
      end
      %~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      function initWfb(obj, init_type)
            %initialize feedback weight matrices of each reservior
            for k = 1:length(obj.architecture.numNodes)
                obj.W_fb{k} = zeros(obj.architecture.numNodes(k), obj.architecture.outputDim);
            end      
      end
      %~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      function initW(obj, init_type, topology)
            %rng('shuffle');
            % generate connecting weight in reservoir
            switch (init_type)
                case 'rand'
                    obj.W = sprand(obj.totalStates, obj.totalStates, obj.hParameters.connectivity);
                    obj.W(obj.W ~= 0) = obj.W(obj.W ~= 0) - 0.5;
                case 'randn'
                    obj.W = sprandn(obj.totalStates, obj.totalStates, obj.hParameters.connectivity);
            end
            % set to zeros those elements that correspond internal weights
            % of each reservoir
            index = 0;
            for k = 1:length(obj.architecture.numNodes)
                obj.W(index+1:index + obj.architecture.numNodes(k),...
                    index+1:index + obj.architecture.numNodes(k)) = 0;
                index = index + obj.architecture.numNodes(k);
            end
            % compute spectral radius i.e. the largest absolute eigen value 
            opts.tol = 1e-3;
            maxVal = max(abs(eigs(obj.W, 1, 'lm', opts)));
            if(maxVal ~= 0)
                obj.W = obj.hParameters.radius * obj.W/maxVal; % normalize W
            end
            %obj.W(logical(eye(size(obj.W)))) = 1;
      end           
      
      %~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      function X = forward(obj, u) % Calculates next step of the system
          %Run each reservoir separately 
          index = 0;
          X_prev = obj.X_last;
          for k = 1:length(obj.architecture.numNodes) % collect states from all reservoirs
             obj.X_last(index+1:index + obj.architecture.numNodes(k)) =...
                 obj.reservoirs(k).forward(obj.W_in{k} * [1; u] + obj.W_fb{k} * obj.Y_last...
                 + obj.W(index+1:index + obj.architecture.numNodes(k), :) * X_prev);
             index = index + obj.architecture.numNodes(k);
          end
          
          obj.X_last = (1 - obj.hParameters.leakage) * X_prev  +...
                                    obj.hParameters.leakage * obj.X_last;
%           index = 0;                      
%           for k = 1:length(obj.architecture.numNodes) % collect states from all reservoirs
%              obj.reservoirs(k).setStates(obj.X_last(index+1:index + obj.architecture.numNodes(k)));
%              index = index + obj.architecture.numNodes(k);
%           end
          X = obj.X_last;
      end
      %~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      function y = evaluate(obj, u) % Evaluates output the system
            X = forward(obj, u);
            obj.Y_last =  obj.W_out * [X; 1; u];     
            y = obj.Y_last;
      end
      %~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      function Y = generate(obj, input, gen_length, feedback_scaling)
          Y = zeros(size(input,1), gen_length);
          Y(:, 1) = feedback_scaling * evaluate(obj, input(:, 1));
          for k = 2:gen_length
               Y(:, k) = feedback_scaling * evaluate(obj, Y(:, k - 1));
          end
      end
      %~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      function Y = predict(obj, input, feedback_scaling)
          Y = zeros(size(input,1), size(input,2));
          for k = 1:size(input,2)
               Y(:, k) = feedback_scaling * evaluate(obj, input(:, k));
          end
      end
      %~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      function setInitStates(obj)
          for k = 1:length(obj.architecture.numNodes)
            obj.reservoirs(k).setInitStates();
          end
      end
      %~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      function resetInitStates(obj)
          for k = 1:length(obj.architecture.numNodes)
            obj.reservoirs(k).resetInitStates();
          end
      end
      %~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      function error = backward(obj, x)

      end       
    end
end