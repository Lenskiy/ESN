classdef Network < handle
    properties (SetAccess = private)
       networkArchitecture;
       Weights;
       blocksInds;
       states;
       layers;
       actFuncs;
       leakage;
       noiseStd;
       indToID;
       IDtoInd;
       savedStates;
       propogationOrder;
    end
    
    properties
    end
    % add methods to work with linked links
    methods
        %------------------------------------------------------------------------------------
        function obj = Network()
            networkArchitecture(1,1) = struct('type', [], 'connectivity', []);
            obj.networkArchitecture = networkArchitecture;
            obj.Weights = sparse([]);
            obj.states = []; % the first index resereved for bias i.e. always 1;
            obj.leakage = [];
            obj.noiseStd = [];
            obj.layers = [];
            obj.indToID = [];
            obj.IDtoInd = 0;
            obj.propogationOrder = [];
          end
        %------------------------------------------------------------------------------------
        function id = addLayer(obj,  numNodes, layerType, params)
            if(isempty(obj.layers))
                id = 1;
            else
                id = obj.layers{end}.id + 1;
            end
            obj.indToID = [obj.indToID id];
            obj.IDtoInd(id) = find(obj.indToID == id);
            obj.blocksInds(obj.IDtoInd(id), :) = [size(obj.Weights,1) + 1, size(obj.Weights,1) + numNodes];
            obj.Weights(end+1:end+numNodes,end+1:end+numNodes) = 0;
            obj.layers{obj.IDtoInd(id)}  = (struct('id', id,...
                                    'numNodes', numNodes,...
                                    'layerType', layerType,... 
                                    'params', params));

            obj.states = [obj.states; zeros(numNodes, 1)];
            obj.leakage = [obj.leakage; params.leakage * ones(numNodes, 1)];
            obj.setActiveFn(id, params.nodeType);
            obj.initLayer(id);
            
            obj.networkArchitecture(obj.IDtoInd(id), obj.IDtoInd(id)) = struct('type', [], 'connectivity', []);
        end
        %------------------------------------------------------------------------------------
        function removeLayer(obj,  id) % broken, lot of works needs to be done
                    assert(ismember(id, obj.indToID), ['ID: ', num2str(id), ' is not found']);
  
                    ind = obj.IDtoInd(id); % could be speeded up
                    obj.indToID(ind) = [];
                    obj.IDtoInd(id:end) = obj.IDtoInd(id:end) - 1; 
                    obj.IDtoInd(id) = 0;
                    obj.layers(ind:end - 1) = obj.layers(ind+1:end);
                    obj.layers(end) = [];
                    inds = obj.blocksInds(ind, :);
                    if(ind < size(obj.blocksInds,1))
                        obj.blocksInds(ind+1:end, :) = obj.blocksInds(ind , 1) + obj.blocksInds(ind + 1:end, :) - obj.blocksInds(ind , 2) - 1;
                    end
                    obj.blocksInds(ind, :) = [];
                    obj.Weights(inds(1):inds(2), :) = [];
                    obj.Weights(:, inds(1):inds(2)) = [];
                    obj.states(inds(1):inds(2)) = [];
                    obj.leakage(inds(1):inds(2)) = [];
                    % remove connections in networkArchitecture
                    obj.networkArchitecture(ind, :) = [];
                    obj.networkArchitecture(:, ind) = [];
                    %update the propagation path
                    obj.propogationOrder = obj.buildPropogationRoute(obj.getConnectivityMatrix(), obj.getIdByName('input'), obj.getIdByName('output'));
        end
        %------------------------------------------------------------------------------------
        function initLayer(obj, id)
            layerIndex = obj.IDtoInd(id);
            switch obj.layers{layerIndex}.layerType
                case 'input'
                    %obj.leakage(obj.getInds(id)) = 1;
                    obj.Weights(obj.getInds(id), obj.getInds(id)) = 1;
                case 'output'
                        ;
                case 'reservoir'
                        obj.initReservoir(id);
                case 'bias'
                    obj.leakage(obj.getInds(id)) = 1;
                    obj.states(obj.getInds(id)) = 1;
                    obj.Weights(obj.getInds(id), obj.getInds(id)) = 1;
            end
        end
        %------------------------------------------------------------------------------------
        function initReservoir(obj, id)
            layerIndex = obj.IDtoInd(id);
            blockSize = obj.blocksInds(layerIndex,2) - obj.blocksInds(layerIndex,1) + 1; % make bocksInd part of layer structure
            %generate connecting weight in reservoir
            switch (obj.layers{layerIndex}.params.initType)
                case 'rand'
                    W = sprand(blockSize, blockSize, obj.layers{layerIndex}.params.connectivity);
                    W(W ~= 0) = W(W ~= 0) - 0.5;
                case 'randn'
                %otherwise
                    W = sprandn(blockSize, blockSize, obj.layers{layerIndex}.params.connectivity);
            end
            obj.setWeights(id, id, W);
            obj.setRadius(id, obj.layers{layerIndex}.params.radius);         
        end 
        %------------------------------------------------------------------------------------
        function setLayerParams(obj, id, params)
            layerIndex = obj.IDtoInd(id);
            names = fieldnames(params);
            for k = 1:length(names)
                switch names{k}
                    case 'nodeType'
                        obj.setActiveFn(id, params.nodeType)
                    case 'radius'
                        obj.setRadius(id, params.radius);
                    case  'leakage'
                        obj.setLeakage(id, params.leakage);
                    case 'connectivity'
                        ;
                    case 'initType'
                        obj.layers{layerIndex}.params.initType = params.initType;
                        obj.initLayer(id);
                end
            end
        end
        %------------------------------------------------------------------------------------
        function setConnectivity(obj, connectivity)
%             delta = obj.params.connectivity - connectivity;
%             if (delta > 0)
%                 inds = find(obj.Weights == 0);
%                 
%             else
%                 
%             end
        end
        %------------------------------------------------------------------------------------
        function setLeakage(obj, id, leakage)
            layerIndex = obj.IDtoInd(id);
            obj.layers{layerIndex}.params.leakage = leakage;
            inds = obj.blocksInds(layerIndex,1):obj.blocksInds(layerIndex,2);
            obj.leakage(inds) = leakage;
        end
        %------------------------------------------------------------------------------------
        function setActiveFn(obj, id, nodeType)
            layerIndex = obj.IDtoInd(id);
            obj.layers{layerIndex}.params.nodeType = nodeType;
        	switch(nodeType)
                case 'linear'
                    obj.layers{layerIndex}.actFunc = @(x) (x);
                case 'tanh'
                    obj.layers{layerIndex}.actFunc = @(x) tanh(x);
                case 'lu'   
                    obj.layers{layerIndex}.actFunc = @(x) max(0, x);
                otherwise
                    obj.layers{layerIndex}.actFunc = @(x) tanh(x);
            end
        end
        %------------------------------------------------------------------------------------
        function setRadius(obj, id, radius)
            layerIndex = obj.IDtoInd(id);
            obj.layers{layerIndex}.params.radius = radius;
            inds = obj.blocksInds(layerIndex,1):obj.blocksInds(layerIndex,2);
            opts.tol = 1e-3;
            maxVal = max(abs(eigs(obj.Weights(inds,inds), 1, 'lm', opts)));
            obj.Weights(inds, inds) = radius * obj.Weights(inds,inds)/maxVal;
        end
        %------------------------------------------------------------------------------------
        function setConnections(obj, arch, type)
            [src, dst]= find(arch ~= 0);
            for k = 1:length(src);
                    params = struct('type', type, 'connectivity', arch(src(k), dst(k)));
                    obj.setConnection(src(k), dst(k), params);
            end
        end
        %------------------------------------------------------------------------------------
        function setConnection(obj, sourceId, destId, params)
            srcInd = obj.IDtoInd(sourceId);
            dstInd = obj.IDtoInd(destId);
            assert(~isempty(srcInd), ['ID: ', num2str(srcInd), ' is not found']);
            assert(~isempty(dstInd), ['ID: ', num2str(dstInd), ' is not found']);
            obj.networkArchitecture(srcInd, dstInd) = params;
            srcInds = obj.blocksInds(srcInd,1):obj.blocksInds(srcInd,2);
            destInds = obj.blocksInds(dstInd, 1):obj.blocksInds(dstInd, 2);
            switch params.type
                case 'rand'
                     W = sprand(length(destInds), length(srcInds),  params.connectivity);
                     W(W ~= 0) = W(W ~= 0) - 0.5;
                case 'randn'
                     W = sprandn(length(destInds), length(srcInds), params.connectivity);
            end
            obj.Weights(destInds, srcInds) = W;
            
            %update the propagation path
            obj.propogationOrder = obj.buildPropogationRoute(obj.getConnectivityMatrix(), obj.getIdByName('input'), obj.getIdByName('output'));           
        end
        %------------------------------------------------------------------------------------
        function visualize(obj)
            conMat = obj.getConnectivityMatrix();
            graph = biograph(conMat);
            graph.NodeAutoSize = 'off';
            graph.ArrowSize = 5;

            for k = length(obj.layers):-1:1
                switch(obj.layers{k}.layerType)
                    case 'input'
                        graph.Nodes(k).Color = [0.7 0.7 0.5];
                        graph.Nodes(k).shape = 'parallelogram';
                        graph.Nodes(k).Label = [num2str(obj.layers{k}.id), ':',obj.layers{k}.layerType];
                        graph.Nodes(k).Size = [50, 15];
                    case 'output'
                        graph.Nodes(obj.layers{k}.id).Color = [0.7 0.7 0.5];
                        graph.Nodes(k).shape = 'parallelogram';
                        graph.Nodes(k).Label = [num2str(obj.layers{k}.id), ':',obj.layers{k}.layerType];
                        graph.Nodes(k).Size = [50, 15];
                    case 'reservoir'
                        graph.Nodes(k).shape = 'circle';
                        graph.Nodes(k).LineWidth = 10 * obj.layers{k}.params.connectivity;
                        numIntStates = obj.blocksInds(end,2) -  obj.blocksInds(1,1); % *needs take into account deleted layers
                        graph.Nodes(k).Size = round(80 * [obj.layers{k}.numNodes; obj.layers{k}.numNodes]/numIntStates) + 20;
                        graph.Nodes(k).Label = [num2str(obj.layers{k}.id), ':',obj.layers{k}.layerType];
                    case 'bias'
                        graph.Nodes(k).shape = 'box';
                        graph.Nodes(k).Size = [15; 15];
                        graph.Nodes(k).Label = num2str(obj.layers{k}.id);
%                         pos = graph.Nodes(obj.IDtoInd(obj.getIdByName('input'))).Position;
%                         graph.Nodes(k).Position = [pos(1), pos(2) + 10];
                end  
                
            end
            
            for k = 1:length(graph.Edges)
                if(isempty(strfind(graph.Edges(k).ID, ['Node ', num2str(obj.IDtoInd(obj.getIdByName('bias')))])))
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
            ind = obj.IDtoInd(obj.getIdByName('output'));
            y = obj.states(obj.blocksInds(ind, 1):obj.blocksInds(ind, 2));
        end
        %-------------------------------------------------------------------------------------
        function setInput(obj, u)
            ind = obj.IDtoInd(obj.getIdByName('input')); % Here is a bottle neck 
            obj.states(obj.blocksInds(ind, 1):obj.blocksInds(ind, 2)) = u;
        end
        %-------------------------------------------------------------------------------------
        function y = predict(obj, input)
            y = zeros(1, length(input));
            for k = 1 : length(input)
                obj.forward(input(k));
                y(k) = obj.getOutput();
            end
        end    
        %-------------------------------------------------------------------------------------
        function y = generate(obj, input, genLenght)
            
            if(nargin() == 2)
                genLenght = length(input);
            end
            
            y = zeros(1, genLenght);
            obj.forward(input(:,1));
            
            y(1) = obj.getOutput();
            for k = 2 : genLenght
                obj.forward(y(k - 1));
                y(k) = obj.getOutput();
            end
        end
       %-------------------------------------------------------------------------------------
       function rememberStates(obj)
            obj.savedStates = obj.states;
       end
       %-------------------------------------------------------------------------------------
       function recallStates(obj)
            obj.states = obj.savedStates;
       end
    end
    
    %=====================================================================
    
    
    methods (Hidden = true)
        %-------------------------------------------------------------------------------------
        function W = getWeights(obj, id1, id2)
            ind1 = obj.IDtoInd(id1);
            ind2 = obj.IDtoInd(id2);
            inds1 = obj.blocksInds(ind1,1):obj.blocksInds(ind1,2);
            inds2 = obj.blocksInds(ind2,1):obj.blocksInds(ind2,2);
            W = obj.Weights(inds1, inds2); 
        end
        %-------------------------------------------------------------------------------------
        function x = activateLayer(obj, x, ind)
                x = obj.layers{ind}.actFunc(x); %this part a bit slows down
        end
        %-------------------------------------------------------------------------------------
        function setWeights(obj, id1, id2, W)
            ind1 = obj.IDtoInd(id1);
            ind2 = obj.IDtoInd(id2);
            inds1 = obj.blocksInds(ind1,1):obj.blocksInds(ind1,2);
            inds2 = obj.blocksInds(ind2,1):obj.blocksInds(ind2,2);
            obj.Weights(inds1, inds2) = W;
        end
        %-------------------------------------------------------------------------------------
        function forwardLayer(obj, id)
            inds = obj.getInds(id);
            obj.states(inds) = obj.leakage(inds) .* obj.layers{obj.IDtoInd(id)}.actFunc(obj.Weights(inds, :) * obj.states)...
                                        + (1 - obj.leakage(inds)) .* obj.states(inds);       
        end
        %-------------------------------------------------------------------------------------
        function x = forward(obj, u) % Calculates next step of the system
            obj.setInput(u);
            
            for k = 1:length(obj.propogationOrder)
                  obj.forwardLayer(obj.propogationOrder(k));
            end
            x = obj.states;
        end
     %-------------------------------------------------------------------------------------
       function idList = getIdByName(obj, name)
            idList = [];
            for k = 1:length(obj.layers)
                if(strcmp(obj.layers{k}.layerType, name))
                    idList = [idList, obj.layers{k}.id];
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
       end
       %-------------------------------------------------------------------------------------
       function states = getStates(obj, IDs)
           states = [];
            for k = 1:length(IDs)
                states = [states; obj.states(obj.getInds(IDs(k)))];
            end           
       end
       %-------------------------------------------------------------------------------------
       function setStates(obj, IDs, states)
           begInd = 1;
            for k = 1:length(IDs)
                endInd = begInd + getNumberOfStates(IDs(k)) - 1;
                obj.states(net.getInds(IDs(k))) = states(begInd:endInd); 
            end           
       end
       %-------------------------------------------------------------------------------------
       function numStates = getNumberOfStates(obj, IDs)
           numStates = 0;
            for k = 1:length(IDs)
                numStates = numStates +  obj.layers{obj.IDtoInd(IDs(k))}.numNodes;
            end    
       end
       %-------------------------------------------------------------------------------------
       function propogationRoute = buildPropogationRoute(obj, conMat, inputID, outputID)
            conMat = ceil(conMat);
            startInd = obj.IDtoInd(inputID);
            outputInd = obj.IDtoInd(outputID);
            nodeToBeVisited(1) = startInd;
            propogationRoute = [];
            while(~isempty(nodeToBeVisited))
                curInd = nodeToBeVisited(1);
                nodeToBeVisited(1) = [];
                if(sum(ismember(propogationRoute, curInd)) ~= 0)
                    continue;
                end
                candidates = find(conMat(curInd,:));
                dist = [];
                for k = 1:length(candidates)
                    tempConMat = conMat;
                    tempConMat(curInd, candidates(k)) = 0;
                    % !!! instead of outputNode, search for the closest common (accessable) node
                    dist(k) = graphshortestpath(tempConMat, 2, outputInd);
                end
                
                [sval, sind] = sort(dist);
                % rearrange the nodes to visit first
                nodeToBeVisited = [nodeToBeVisited candidates(sind)];
                propogationRoute = [propogationRoute, curInd];
            end
            propogationRoute = obj.indToID(propogationRoute);
       end    
    end    
end