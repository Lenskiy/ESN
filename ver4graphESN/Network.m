classdef Network < handle
    properties (SetAccess = private)
       networkArchitecture;
       Weights;
       blocksInds;
       
       forwardStates;
       backwardStates;
       
       layersCells; % paramters of every layers
       
       leakage;  % individual leakages per node
       noiseStd;
       
       indToID;
       IDtoInd;
       
       savedStates;
       
       propogationOrder;
       outputIDs;
       inputIDs;
    end
    
    properties
    end

    methods
        %------------------------------------------------------------------------------------
        function obj = Network()
            maxNumCompThreads = 8;
            networkArchitecture(1,1) = struct('initType', [],...
                                              'connectivity', [],...
                                              'radius', []);
            obj.networkArchitecture = networkArchitecture;
            obj.Weights = sparse([]);
            obj.forwardStates = []; % the first index resereved for bias i.e. always 1;
            obj.leakage = ones(2, 1);
            obj.noiseStd = [];
            obj.inputIDs = [];
            obj.outputIDs = [];
            % {id, nunNodes, layerType,  {nodeType, leakageInit, leakageVal}}
            obj.layersCells = {[], [], [], {[], [], []}};                   
            obj.indToID = [];
            obj.IDtoInd = 0;
            obj.propogationOrder = [];
          end
        %------------------------------------------------------------------------------------
        function id = addLayer(obj,  numNodes, layerType, params)
            if(isempty(obj.layersCells{1}))
                id = 1;
            else
                id = obj.layersCells{end, 1} + 1;
            end
            obj.indToID = [obj.indToID id];
            obj.IDtoInd(id) = find(obj.indToID == id);
            obj.blocksInds(obj.IDtoInd(id), :) = [size(obj.Weights,1) + 1, size(obj.Weights,1) + numNodes];
            obj.Weights(end+1:end+numNodes,end+1:end+numNodes) = 0; 

            obj.layersCells(obj.IDtoInd(id), :) = {id, numNodes, layerType, struct2cell(params)};
            
            obj.forwardStates = full([obj.forwardStates; zeros(numNodes, 1)]);
            
            obj.setActiveFn(id, params.nodeType);
            obj.setLeakage(id, params.leakageInit, params.leakageVal);
            obj.networkArchitecture(obj.IDtoInd(id), obj.IDtoInd(id)) =...
                             struct('initType', [],...
                                    'connectivity', [],...
                                    'radius', []);
            obj.initLayer(id);

        end
        %------------------------------------------------------------------------------------
        function removeLayer(obj,  id) % broken, lot of works needs to be done
                    assert(ismember(id, obj.indToID), ['ID: ', num2str(id), ' is not found']);
                    %check if the layer to be removed is an output layer
                    ind = find(obj.outputIDs == id);
                    if(~isempty(ind))
                        obj.outputIDs(ind) = [];
                    end
                    %check if the layer to be removed is an input layer
                    ind = find(obj.inputIDs == id);
                    if(~isempty(ind))
                        obj.inputIDs(ind) = [];
                    end
                    
                    ind = obj.IDtoInd(id); % could be speeded up
                    
                    inds = obj.blocksInds(ind, :); % indexes to be removed from the Weight matrix
                    obj.blocksInds(ind+1:end, :) = obj.blocksInds(ind+1:end, :)  - obj.blocksInds(ind,2) + obj.blocksInds(ind,1) - 1;
                    obj.blocksInds(ind, :) = []; % remove the row with indexes
                    obj.IDtoInd(id:end) = obj.IDtoInd(id:end) - 1; 
                    obj.IDtoInd(id) = 0;                    
                    obj.indToID(ind) = [];                   
                    obj.Weights(:, inds(1):inds(2)) = [];
                    obj.Weights(inds(1):inds(2), :) = [];
                    obj.forwardStates(inds(1):inds(2)) = [];
                    obj.leakage(inds(1):inds(2)) = [];
                    % remove connections in networkArchitecture
                    obj.networkArchitecture(ind, :) = [];
                    obj.networkArchitecture(:, ind) = [];
                    
                    obj.layersCells(ind, :) = [];
                    %update the propagation path
                    obj.propogationOrder = obj.buildPropogationRoute(obj.getConnectivityMatrix(),...
                                                obj.getIdByType('input'),...
                                                obj.getIdByType('output'));
        end
        %------------------------------------------------------------------------------------
        function initLayer(obj, id)
            %inds = obj.layersStructs(obj.IDtoInd(id)).inds;
            layerIndex = obj.IDtoInd(id);
            inds = obj.blocksInds(layerIndex, 1) : obj.blocksInds(layerIndex, 2);
            switch  obj.layersCells{layerIndex, 3} % obj.layersStructs(layerIndex).layerType
                case 'input'
                    %obj.leakage(obj.getInds(id)) = 1;
                    obj.Weights(inds, inds) = eye(obj.layersCells{layerIndex, 2}); %eye(obj.layersStructs(id).numNodes);
                    obj.inputIDs = [obj.inputIDs id];
                case 'output'
                    obj.outputIDs = [obj.outputIDs id];
                case 'layer' % for the reser
                    obj.Weights(inds, inds) = eye(obj.layersCells{layerIndex, 2}); %eye(obj.layersStructs(id).numNodes);
                case 'bias'
                    %obj.leakage(obj.getInds(id)) = 1;
                    obj.setStates(obj.getInds(id), 1);
                    obj.Weights(inds, inds) = 1;
                    obj.setLeakage(id, 'constant', '1');
            end
        end
        %------------------------------------------------------------------------------------
        function setLayerParams(obj, id, params)
            names = fieldnames(params);
            for k = 1:length(names)
                switch names{k}
                    case 'nodeType'
                        obj.setActiveFn(id, params.nodeType)
                    case 'radius'
                        obj.setRadius(id, params.radius);
                    case  'leakage'
                        obj.setLeakage(id, params.leakageInit, params.leakageVal);
                    case 'connectivity'
                        obj.setConnectivity(id, params.connectivity);
%                     case 'initType'
%                         obj.networkArchitecture(layerIndex, layerIndex).initType = params.initType;
                end
            end
        end
        %------------------------------------------------------------------------------------
        function setConnectivity(obj, id, connectivity)
            if(length(id) == 1)
                id1 = id;
                id2 = id1;
            else
                id1 = id(1);
                id2 = id(2);
            end

            assert(obj.IDtoInd(id1) ~= 0, ['ID: ', num2str(id1), ' is not found']);
            assert(obj.IDtoInd(id2) ~= 0, ['ID: ', num2str(id2), ' is not found']);
            
            delta = connectivity - obj.networkArchitecture(obj.IDtoInd(id1), obj.IDtoInd(id2)).connectivity;
            
            W = obj.getWeights(id1, id2);

            if (delta > 0) % add more weights 
                inds = find(obj.Weights == 0);
                zeroElementsInds = randperm(length(inds), floor(length(inds) * delta));
                switch(obj.networkArchitecture(obj.IDtoInd(id1), obj.IDtoInd(id2)).initType)
                    case 'rand'
                        W(inds(zeroElementsInds)) = rand(length(zeroElementsInds), 1);
                        W(inds(zeroElementsInds)) = obj.W(inds(zeroElementsInds)) - 0.5;
                    case 'randn'
                        W(inds(zeroElementsInds)) = randn(length(zeroElementsInds), 1);
                end
            else    % remove some weights
                inds = find(W ~= 0);
                nonZeroElementsInds = randperm(length(inds), floor(length(inds) * abs(delta)));
                switch(obj.networkArchitecture(obj.IDtoInd(id1), obj.IDtoInd(id2)).initType)
                    case 'rand'
                        W(inds(nonZeroElementsInds)) = 0;
                    case 'randn'
                        W(inds(nonZeroElementsInds)) = 0;
                end                
            end
            obj.setWeights(id1, id2, W);
        end
        %------------------------------------------------------------------------------------
        function connectivity = getConnectivity(obj, id)
            if(length(id) == 1)
                id1 = id;
                id2 = id1;
            else
                id1 = id(1);
                id2 = id(2);
            end
            assert(obj.IDtoInd(id1) ~= 0, ['ID: ', num2str(id1), ' is not found']);
            assert(obj.IDtoInd(id2) ~= 0, ['ID: ', num2str(id2), ' is not found']);
            connectivity = obj.networkArchitecture(obj.IDtoInd(id1), obj.IDtoInd(id2)).connectivity;
        end
        %------------------------------------------------------------------------------------   
        function setLeakage(obj, id, type, val)
            layerInd = obj.IDtoInd(id);
            inds = obj.blocksInds(layerInd,1):obj.blocksInds(layerInd,2);
            switch (type)
                case 'constant'
                    obj.leakage(inds) = str2num(val) * ones(obj.layersCells{obj.IDtoInd(id), 2}, 1);
                    %obj.layersCells{obj.IDtoInd(id), 4}{5} = str2num(val) * ones(obj.layersCells{obj.IDtoInd(id), 2}, 1);
                case 'rand'
                    vals = split(val, ':');
                    minVal = str2num(vals{1});
                    maxVal = str2num(vals{2});
                    obj.leakage(inds) = (maxVal - minVal) * rand(obj.layersCells{obj.IDtoInd(id), 2}, 1) + minVal;
                    %obj.layersCells{obj.IDtoInd(id), 4}{5} = (maxVal - minVal) * rand(obj.layersCells{obj.IDtoInd(id), 2}, 1) + minVal;
                otherwise
                    obj.leakage(inds) = str2num(val) * ones(obj.layersCells{obj.IDtoInd(id), 2}, 1);
                    %obj.layersCells{obj.IDtoInd(id), 4}{5} = str2num(val) * ones(obj.layersCells{obj.IDtoInd(id), 2}, 1);
                    
            end
            %obj.layersCells{obj.IDtoInd(id), 4}{2} = leakage;
            %inds = obj.blocksInds(layerIndex,1):obj.blocksInds(layerIndex,2);
            %obj.leakage(inds) = leakage;
        end
        %------------------------------------------------------------------------------------
        function setActiveFn(obj, id, nodeType)
            layerIndex = obj.IDtoInd(id);
            switch(nodeType)
            case 'linear'
                obj.layersCells{layerIndex, 4}{1} = @(x) (x);
                %obj.actFunc(inds) = @(x) (x);
            case 'tanh'
                obj.layersCells{layerIndex, 4}{1} = @(x) tanh(x);
                %obj.actFunc(inds) = @(x) tanh(x);
            case 'lu'   
                obj.layersCells{layerIndex, 4}{1} = @(x) max(0, x);
                %obj.actFunc(inds) = @(x) max(0, x);
                otherwise
                obj.layersCells{layerIndex, 4}{1} = @(x) tanh(x);
                %obj.actFunc(inds) = @(x) tanh(x);
            end
        end
        %------------------------------------------------------------------------------------
        function radius = getRadius(obj, id)
            if(length(id) == 1)
                id1 = id;
                id2 = id1;
            else
                id1 = id(1);
                id2 = id(2);
            end
            assert(obj.IDtoInd(id1) ~= 0, ['ID: ', num2str(id1), ' is not found']);
            assert(obj.IDtoInd(id2) ~= 0, ['ID: ', num2str(id2), ' is not found']);            
            radius = obj.networkArchitecture(obj.IDtoInd(id1), obj.IDtoInd(id2)).radius;
        end
        %------------------------------------------------------------------------------------            
        function setRadius(obj, radius, id)   
            if(length(id) == 1)
                id1 = id;
                id2 = id1;
            else
                id1 = id(1);
                id2 = id(2);
            end
            layerIndex1 = obj.IDtoInd(id1);
            layerIndex2 = obj.IDtoInd(id2);
            assert(layerIndex1 ~= 0, ['ID: ', num2str(id1), ' is not found']);
            assert(layerIndex2 ~= 0, ['ID: ', num2str(id2), ' is not found']);
            
            W = obj.getWeights(layerIndex1, layerIndex2);
            if(size(W, 1) == size(W, 2))
            %assert(, ['A weight matrix of size ', num2str(size(W,1)), ' by ', num2str(size(W,2)) , ' is not squre']); 
                opts.tol = 1e-3;
                maxVal = max(abs(eigs(W, 1, 'lm', opts)));
            else
                maxVal = 1;
            end
            obj.setWeights(id1, id2, radius * W/maxVal);
        end
        %------------------------------------------------------------------------------------
        function setConnections(obj, arch, initTypes, radii)
            [src, dst]= find(arch ~= 0);
            for k = 1:length(src)
                    obj.setConnection(...
                            src(k),... 
                            dst(k),... 
                            struct('initType',      initTypes{k},...
                                   'connectivity',  arch(src(k), dst(k)),...
                                   'radius',        radii(k))...
                               );
            end
        end
        %------------------------------------------------------------------------------------
        function setConnection(obj, sourceId, destId, params)
            srcInd = obj.IDtoInd(sourceId);
            dstInd = obj.IDtoInd(destId);
            assert(~isempty(srcInd), ['ID: ', num2str(srcInd), ' is not found']);
            assert(~isempty(dstInd), ['ID: ', num2str(dstInd), ' is not found']);
            
            if(~isfield(params, 'radius'))
                params.radius = [];
            end
            obj.networkArchitecture(srcInd, dstInd) = params;

            srcInds  = obj.blocksInds(srcInd, 1):obj.blocksInds(srcInd, 2);
            destInds = obj.blocksInds(dstInd, 1):obj.blocksInds(dstInd, 2);
            nCols = numel(destInds);
            nRows = numel(srcInds);

            switch params.initType
                case 'rand'
                    nonZeroElementsInds = randperm(nRows * nCols,...
                        floor(nRows * nCols * (1 - params.connectivity)));
                    W = rand(nCols, nRows);
                    W(W ~= 0) = W(W ~= 0) - 0.5;
                    W(nonZeroElementsInds) = 0;                                         
                case 'randn'
                    nonZeroElementsInds = randperm(nRows * nCols,...
                        floor(nRows * nCols * (1 - params.connectivity)));
                    W = randn(nCols, nRows);
                    W(nonZeroElementsInds) = 0;  
            end
           % obj.Weights(srcInds, destInds) = W;
            obj.setWeights(sourceId, destId, W);
            if(~isempty(params.radius))
                obj.setRadius(params.radius, [sourceId, destId]);
            end
            %update the propagation path
            obj.propogationOrder = obj.buildPropogationRoute(obj.getConnectivityMatrix(), obj.getIdByType('input'), obj.getIdByType('output'));           
        end
        %------------------------------------------------------------------------------------
        function rmConnection(obj, sourceId, destId)
            setConnection(obj, sourceId, destId, struct('type', 'randn', 'connectivity', 0.0))
        end
        %------------------------------------------------------------------------------------
        function visualize(obj)
            conMat = obj.getConnectivityMatrix();
            graph = biograph(conMat);
            graph.NodeAutoSize = 'off';
            graph.ArrowSize = 5;
            maxReservoirSize = 0;
            for k = 1:size(obj.layersCells,1) %size(obj.layersStructs,2)
                if(obj.networkArchitecture(k, k).connectivity ~= 0)
                    maxReservoirSize = max(maxReservoirSize, obj.layersCells{k, 2});
                end
            end

            for k = size(obj.layersCells,1):-1:1
                switch(obj.layersCells{k, 3})
                    case 'input'
                        graph.Nodes(k).Color = [0.7 0.7 0.5];
                        graph.Nodes(k).shape = 'parallelogram';
                        graph.Nodes(k).Label = [num2str(obj.layersCells{k, 1}), ':', obj.layersCells{k, 3}(1:3) '(',num2str(obj.layersCells{k, 2}),')'];
                        graph.Nodes(k).Size = [50, 15];
                    case 'output'
                        graph.Nodes(obj.IDtoInd(obj.layersCells{k, 1})).Color = [0.7 0.7 0.5]; 
                        graph.Nodes(k).shape = 'parallelogram';
                        graph.Nodes(k).Label = [num2str(obj.layersCells{k, 1}), ':', obj.layersCells{k,3}(1:3) '(',num2str(obj.layersCells{k, 2}),')'];
                        graph.Nodes(k).Size = [50, 15];
                    case 'layer'
                        if(obj.networkArchitecture(k, k).connectivity ~= 0)
                            graph.Nodes(k).shape = 'circle';
                            graph.Nodes(k).LineWidth = 10 * obj.networkArchitecture(k, k).connectivity;
                            graph.Nodes(k).Size = round(80 * [obj.layersCells{k, 2}; obj.layersCells{k, 2}]/maxReservoirSize) + 20;
                        else
                            graph.Nodes(k).shape = 'box';
                            graph.Nodes(k).Size = round(80 * [obj.layersCells{k, 2}; 0.25*obj.layersCells{k, 2}]/obj.layersCells{k, 2}) + 20;
                        end
                            graph.Nodes(k).Label = [num2str(obj.layersCells{k, 1}), ':', obj.layersCells{k, 3}(1:3), '(',num2str(obj.layersCells{k, 2}),')'];   
                            vals = split(obj.layersCells{k, 4}{3}, ':');
                            graph.Nodes(k).Color = str2num(vals{1}) * graph.Nodes(k).Color;   
                    case 'bias'
                        graph.Nodes(k).shape = 'box';
                        graph.Nodes(k).Size = [15; 15];
                        graph.Nodes(k).Label = num2str(obj.layersCells{k, 1});
                end  
                
            end
            
            for k = 1:length(graph.Edges)
                if(isempty(strfind(graph.Edges(k).ID, ['Node ', num2str(obj.IDtoInd(obj.getIdByType('bias')))])))
                    graph.Edges(k).LineWidth = 10 * graph.Edges(k).Weight;
                else
                    graph.Edges(k).LineWidth = 0.1;
                end
            end
            

            h = view(graph);
        end
        %------------------------------------------------------------------------------------
        function mat = getConnectivityMatrix(obj)
            dim = size(obj.networkArchitecture, 1);
            mat = sparse(dim, dim);
            for k = 1:dim
                for l = 1:dim
                    if(~isempty(obj.networkArchitecture(k,l).connectivity))
                        mat(k, l) = obj.networkArchitecture(k,l).connectivity;
                    end
                end
            end
        end
        %-------------------------------------------------------------------------------------
        function y = getOutput(obj)
            ind = obj.IDtoInd(obj.outputIDs); 
            y = obj.forwardStates(obj.blocksInds(ind, 1):obj.blocksInds(ind, 2));
        end
        %-------------------------------------------------------------------------------------
        function setInput(obj, u)
           for ind = obj.IDtoInd(obj.inputIDs)
               inds = obj.blocksInds(ind, 1):obj.blocksInds(ind, 2);
               obj.forwardStates(inds) = u; % u(inds - 1); %  - 1 because bias should always 1;
           end     
        end
        %-------------------------------------------------------------------------------------
        function y = predict(obj, input)
            dimOutput = sum(obj.layersCells{obj.IDtoInd(obj.outputIDs), 2});
            y = zeros(dimOutput, size(input,2));
            for k = 1 : size(input,2)
                obj.forward(input(:,k));
                y(:,k) = obj.getOutput();
            end
        end    
        %-------------------------------------------------------------------------------------
        function y = generate(obj, input, genLenght)
            
            if(nargin() == 2)
                genLenght = size(input,2);
            end
            
            y = zeros(size(input,1), genLenght);
            obj.forward(input(:,1));
            
            y(:, 1) = obj.getOutput();
            for k = 2 : genLenght
                obj.forward(y(:, k - 1));
                y(:, k) = obj.getOutput();
            end
        end
       %-------------------------------------------------------------------------------------
       function rememberStates(obj)
            obj.savedStates = obj.forwardStates;
       end
       %-------------------------------------------------------------------------------------
       function recallStates(obj)
            obj.forwardStates = obj.savedStates;
       end
    end
    
    %=====================================================================
    
    
    methods (Hidden = true)
        %-------------------------------------------------------------------------------------
        function W = getWeights(obj, id1, id2)
            layerIndex = obj.IDtoInd(id1);
            inds1 = obj.blocksInds(layerIndex, 1) : obj.blocksInds(layerIndex, 2);  
            layerIndex = obj.IDtoInd(id2);
            inds2 = obj.blocksInds(layerIndex, 1) : obj.blocksInds(layerIndex, 2);              
            W = obj.Weights(inds2, inds1); 
        end
        %-------------------------------------------------------------------------------------
        function x = activateLayer(obj, x, ind) % this function is not used
                x = obj.layersCells{ind, 4}{2}(x); %this part a bit slows down
        end
        %-------------------------------------------------------------------------------------
        function setWeights(obj, id1, id2, W)
            layerIndex = obj.IDtoInd(id1);
            inds1 = obj.blocksInds(layerIndex, 1) : obj.blocksInds(layerIndex, 2);  
            layerIndex = obj.IDtoInd(id2);
            inds2 = obj.blocksInds(layerIndex, 1) : obj.blocksInds(layerIndex, 2);  
            obj.Weights(inds2, inds1) = W;
        end
        %-------------------------------------------------------------------------------------
        function setWeightsForSelectedWeights(obj, IDs, ID, W)
            obj.blocksInds(obj.IDtoInd(IDs), :);
            selectedInds = [0; cumsum(obj.blocksInds(obj.IDtoInd(IDs), 2) -  obj.blocksInds(obj.IDtoInd(IDs), 1) + 1)];
            for k = 1:numel(IDs)
                obj.setWeights(IDs(k), ID, W(:, selectedInds(k) + 1:selectedInds(k+1)));
            end
        end        
        %-------------------------------------------------------------------------------------
        function forwardLayer(obj, id)
            layerInd = obj.IDtoInd(id);
            inds = obj.blocksInds(layerInd,1):obj.blocksInds(layerInd,2);
       
            %leakage = obj.layersCells{layerInd, 4}{4};
            lk = obj.leakage(inds);
            obj.forwardStates(inds) =  lk .* obj.layersCells{layerInd, 4}{1}(obj.Weights(inds, :) * obj.forwardStates)...
                                        + (1 - lk) .* obj.forwardStates(inds); 
                                   
        end
        %-------------------------------------------------------------------------------------
        function backwardLayer(obj, id)
            layerInd = obj.IDtoInd(id);
            inds = obj.blocksInds(layerInd,1):obj.blocksInds(layerInd,2);
       
            %leakage = obj.layersCells{layerInd, 4}{4};
            lk = obj.leakage(inds);
            obj.forwardStates(inds) =  lk .* obj.layersCells{layerInd, 4}{1}(obj.Weights(inds, :) * obj.forwardStates)...
                                        + (1 - lk) .* obj.forwardStates(inds); 
                                    
            %obj.backwardStates(inds) = 
                                   
        end        
        %-------------------------------------------------------------------------------------
        function forward(obj, u) % Calculates next step of the system
            obj.setInput(u);
            
            for k = 1:numel(obj.propogationOrder)
                  obj.forwardLayer(obj.propogationOrder(k));
            end
            %x = obj.forwardStates; %remove this, to speed up
        end
     %-------------------------------------------------------------------------------------
       function idList = getIdByType(obj, name)
            idList = [];
            for k = 1:numel(obj.indToID)
                if(strcmp(obj.layersCells{k, 3}, name))
                    idList = [idList, obj.layersCells{k, 1}];
                end
            end
       end
       %------------------------------------------------------------------------------------- 
       function idConTo = getNextNodes(obj, id)
             conMat = obj.getConnectivityMatrix();
             indConTo = conMat(obj.IDtoInd(id), :) ~= 0;
             idConTo = obj.indToID(indConTo);
       end
       %------------------------------------------------------------------------------------- 
       function idConTo = getPrevNodes(obj, id)
             conMat = obj.getConnectivityMatrix();
             indConTo = conMat(:,obj.IDtoInd(id)) ~= 0;
             idConTo = obj.indToID(indConTo);
       end       
       %------------------------------------------------------------------------------------- 
       function stateInds = getInds(obj, id)
                indLayer = obj.IDtoInd(id);
                stateInds = obj.blocksInds(indLayer,1):obj.blocksInds(indLayer,2);
                %stateInds = obj.layersStructs(obj.IDtoInd(id)).inds;
       end
       %-------------------------------------------------------------------------------------
       function states = getStates(obj, IDs)
            states = zeros(1, sum(obj.blocksInds(obj.IDtoInd(IDs), 2) - obj.blocksInds(obj.IDtoInd(IDs), 1) + 1));
            curInd = 0;
            for ind = obj.IDtoInd(IDs)
                numNodes = obj.blocksInds(ind, 2) - obj.blocksInds(ind, 1) + 1;
                states(curInd+1:curInd + numNodes) = obj.forwardStates(obj.blocksInds(ind, 1):obj.blocksInds(ind, 2));
                curInd = curInd + numNodes;
            end           
       end
       %-------------------------------------------------------------------------------------
       function setStates(obj, id, states)
           indLayer = obj.IDtoInd(id);
           obj.forwardStates(obj.blocksInds(indLayer,1):obj.blocksInds(indLayer,2)) = states;
       end
       %-------------------------------------------------------------------------------------
       function setLayersStates(obj, IDs, states)
           begInd = 1;
            for k = 1:numel(IDs)
                endInd = begInd + obj.getNumberOfStates(IDs(k)) - 1;
                setStates(obj, IDs(k), states(begInd:endInd))
            end           
       end
       %-------------------------------------------------------------------------------------
       function numStates = getNumberOfStates(obj, IDs)
           numStates = 0;
            for k = 1:numel(IDs)
                numStates = numStates +  obj.layersCells{obj.IDtoInd(IDs(k)), 2}; %obj.layersStructs(obj.IDtoInd(IDs(k))).numNodes;
            end    
       end
       %-------------------------------------------------------------------------------------
       function propogationRoute = buildPropogationRoute(obj, conMat, inputID, outputID)
            % based on Breadth-first search
                conMat = ceil(conMat);
                conMat(logical(eye(size(conMat)))) = 0;
                % if many inputs create a virtual root to witch all inputs
                % are connected
                extConMat = sparse(zeros(size(conMat,1) + 1, size(conMat, 2) + 1));
                extConMat(2:end, 2:end) = conMat;
                extConMat(1,obj.inputIDs + 1) = 1; 
                
                startInd = 1;
                outputInd = obj.IDtoInd(outputID) + 1; % add 1 because of a virtual node
                
                propogationRoute = [];
                
                traversal = graphtraverse(extConMat, startInd, 'Method', 'BFS');

                traversal = setdiff(traversal, [1, obj.inputIDs + 1]); % remove virtual root and the inputs
                
                for k = 1:length(traversal) 
                    if(graphshortestpath(extConMat, traversal(k), outputInd) ~= inf)
                        propogationRoute = [propogationRoute, traversal(k)];
                    end
                end
                
                propogationRoute = propogationRoute - 1; % compensate for virtual root
                
                propogationRoute = obj.indToID(propogationRoute);
       end  
       %-------------------------------------------------------------------------------------
       function propogationOrder = getPropogationRoute(obj)
           propogationOrder = [obj.getIdByType('input'), obj.propogationOrder];
       end
       %-------------------------------------------------------------------------------------
       function propogationOrder = getBackPropogationRoute(obj)
           propogationOrder = [obj.getIdByType('output'), obj.propogationOrder(end:-1:1)]; % add input node
       end       
    end    
end