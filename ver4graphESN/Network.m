classdef Network < handle
    properties (SetAccess = private)
       networkArchitecture;
       Weights;
       blocksInds;
       states;
       layersStructs;
       layersCells;
       
       %leakage;
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
    % add methods to work with linked links
    methods
        %------------------------------------------------------------------------------------
        function obj = Network()
            maxNumCompThreads = 8;
            networkArchitecture(1,1) = struct('type', [], 'connectivity', []);
            obj.networkArchitecture = networkArchitecture;
            obj.Weights = sparse([]);
            obj.states = []; % the first index resereved for bias i.e. always 1;
            obj.noiseStd = [];
            obj.inputIDs = [];
            obj.outputIDs = [];
            obj.layersStructs = (struct('id', {},...
                                    'numNodes', {},...
                                    'layerType', {},...
                                    'actFunc',{},...
                                    'inds',[],...
                                    'params', {}));
            obj.layersCells = [];
            obj.indToID = [];
            obj.IDtoInd = 0;
            obj.propogationOrder = [];
          end
        %------------------------------------------------------------------------------------
        function id = addLayer(obj,  numNodes, layerType, params)
            if(isempty(obj.layersStructs))
                id = 1;
            else
                id = obj.layersStructs(end).id + 1;
            end
            obj.indToID = [obj.indToID id];
            obj.IDtoInd(id) = find(obj.indToID == id);
            obj.blocksInds(obj.IDtoInd(id), :) = [size(obj.Weights,1) + 1, size(obj.Weights,1) + numNodes];
            obj.Weights(end+1:end+numNodes,end+1:end+numNodes) = 0;
            obj.layersStructs(obj.IDtoInd(id))  = (struct('id', id,...
                                    'numNodes', numNodes,...
                                    'layerType', layerType,... 
                                    'actFunc', [],...
                                    'inds',obj.blocksInds(obj.IDtoInd(id), 1):obj.blocksInds(obj.IDtoInd(id), 2),...
                                    'params', params));

            obj.states = full([obj.states, zeros(1, numNodes)]);
            %obj.leakage = [obj.leakage; params.leakage * ones(numNodes, 1)];
            obj.setActiveFn(id, params.nodeType);
            obj.initLayer(id);
            obj.layersCells = struct2cell(obj.layersStructs(obj.indToID));
            obj.networkArchitecture(obj.IDtoInd(id), obj.IDtoInd(id)) = struct('type', [], 'connectivity', []);
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
                    
                    obj.layersStructs(ind:end - 1) = obj.layersStructs(ind+1:end);
                    obj.layersStructs(end) = [];
                    inds = obj.blocksInds(ind, :);
                    if(ind < numel(obj.indToID))
                        obj.blocksInds(ind+1:end, :) = obj.blocksInds(ind , 1) + obj.blocksInds(ind + 1:end, :) - obj.blocksInds(ind , 2) - 1;
                        for k = ind+1:numel(obj.indToID)
                            obj.layersStructs(obj.IDtoInd(id)).inds = obj.blocksInds(k, 1) : obj.blocksInds(k, 2);
                        end
                    end
                    
                    obj.IDtoInd(id:end) = obj.IDtoInd(id:end) - 1; 
                    obj.IDtoInd(id) = 0;                    
                    obj.indToID(ind) = [];
                    obj.blocksInds(ind, :) = [];
                    obj.Weights(inds(1):inds(2), :) = [];
                    obj.Weights(:, inds(1):inds(2)) = [];
                    obj.states(inds(1):inds(2)) = [];
                    %obj.leakage(inds(1):inds(2)) = [];
                    % remove connections in networkArchitecture
                    obj.networkArchitecture(ind, :) = [];
                    obj.networkArchitecture(:, ind) = [];
                    %update the propagation path
                    obj.layersCells = struct2cell(obj.layersStructs(obj.indToID));
                    obj.propogationOrder = obj.buildPropogationRoute(obj.getConnectivityMatrix(), obj.getIdByName('input'), obj.getIdByName('output'));
        end
        %------------------------------------------------------------------------------------
        function initLayer(obj, id)
            layerIndex = obj.IDtoInd(id);
            switch obj.layersStructs(layerIndex).layerType
                case 'input'
                    %obj.leakage(obj.getInds(id)) = 1;
                    inds = obj.layersStructs(obj.IDtoInd(id)).inds;
                    obj.Weights(inds, inds) = eye(obj.layersStructs(id).numNodes);
                    obj.inputIDs = [obj.inputIDs id];
                case 'output'
                    obj.outputIDs = [obj.outputIDs id];
                case 'reservoir'
                    obj.initReservoir(id);
                case 'layer'
                    inds = obj.layersStructs(obj.IDtoInd(id)).inds;
                    obj.Weights(inds, inds) = eye(obj.layersStructs(id).numNodes);
                case 'bias'
                    %obj.leakage(obj.getInds(id)) = 1;
                    obj.setStates(obj.getInds(id), 1);
                    inds = obj.layersStructs(obj.IDtoInd(id)).inds;
                    obj.Weights(inds, inds) = 1;
                    obj.setLeakage(id, 1);
            end
        end
        %------------------------------------------------------------------------------------
        function initReservoir(obj, id)
            layerIndex = obj.IDtoInd(id);
            blockSize = obj.blocksInds(layerIndex,2) - obj.blocksInds(layerIndex,1) + 1; % make bocksInd part of layer structure
            %generate connecting weight in reservoir
            switch (obj.layersStructs(layerIndex).params.initType)
                case 'rand'
                    W = sprand(blockSize, blockSize, obj.layersStructs(layerIndex).params.connectivity);
                    W(W ~= 0) = W(W ~= 0) - 0.5;
                case 'randn'
                %otherwise
                    W = sprandn(blockSize, blockSize, obj.layersStructs(layerIndex).params.connectivity);
            end
            obj.setWeights(id, id, W);
            obj.setRadius(id, obj.layersStructs(layerIndex).params.radius);         
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
                        obj.layersStructs(layerIndex).params.initType = params.initType;
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
            obj.layersStructs(layerIndex).params.leakage = leakage;
            %inds = obj.blocksInds(layerIndex,1):obj.blocksInds(layerIndex,2);
            %obj.leakage(inds) = leakage;
        end
        %------------------------------------------------------------------------------------
        function setActiveFn(obj, id, nodeType)
            layerIndex = obj.IDtoInd(id);
            obj.layersStructs(layerIndex).params.nodeType = nodeType;
        	switch(nodeType)
                case 'linear'
                    obj.layersStructs(layerIndex).actFunc = @(x) (x);
                case 'tanh'
                    obj.layersStructs(layerIndex).actFunc = @(x) tanh(x);
                case 'lu'   
                    obj.layersStructs(layerIndex).actFunc = @(x) max(0, x);
                otherwise
                    obj.layersStructs(layerIndex).actFunc = @(x) tanh(x);
            end
        end
        %------------------------------------------------------------------------------------
        function setRadius(obj, id, radius)
            layerIndex = obj.IDtoInd(id);
            obj.layersStructs(layerIndex).params.radius = radius;
            inds = obj.layersStructs(layerIndex).inds;
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
            srcInds = obj.layersStructs(srcInd).inds;
            destInds = obj.layersStructs(dstInd).inds;
            switch params.type
                case 'rand'
                     W = sprand(numel(srcInds), numel(destInds),  params.connectivity);
                     W(W ~= 0) = W(W ~= 0) - 0.5;
                case 'randn'
                     W = sprandn(numel(srcInds), numel(destInds), params.connectivity);
%                         W = randn(length(destInds), length(srcInds));
            end
            obj.Weights(srcInds, destInds) = W;
            
            %update the propagation path
            obj.propogationOrder = obj.buildPropogationRoute(obj.getConnectivityMatrix(), obj.getIdByName('input'), obj.getIdByName('output'));           
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

            for k = length(obj.layersStructs):-1:1
                switch(obj.layersStructs(k).layerType)
                    case 'input'
                        graph.Nodes(k).Color = [0.7 0.7 0.5];
                        graph.Nodes(k).shape = 'parallelogram';
                        graph.Nodes(k).Label = [num2str(obj.layersStructs(k).id), ':',obj.layersStructs(k).layerType(1:3) '(',num2str(obj.layersStructs(k).numNodes),')'];
                        graph.Nodes(k).Size = [50, 15];
                    case 'output'
                        graph.Nodes(obj.layersStructs(k).id).Color = obj.layersStructs(k).params.leakage * [0.7 0.7 0.5];
                        graph.Nodes(k).shape = 'parallelogram';
                        graph.Nodes(k).Label = [num2str(obj.layersStructs(k).id), ':',obj.layersStructs(k).layerType(1:3) '(',num2str(obj.layersStructs(k).numNodes),')'];
                        graph.Nodes(k).Size = [50, 15];
                    case 'reservoir'
                        graph.Nodes(k).shape = 'circle';
                        graph.Nodes(k).LineWidth = 10 * obj.layersStructs(k).params.connectivity;
                        graph.Nodes(k).Size = round(80 * [obj.layersStructs(k).numNodes; obj.layersStructs(k).numNodes]/obj.layersStructs(k).numNodes) + 20;
                        graph.Nodes(k).Label = [num2str(obj.layersStructs(k).id), ':',obj.layersStructs(k).layerType(1:3), '(',num2str(obj.layersStructs(k).numNodes),')'];
                        graph.Nodes(obj.layersStructs(k).id).Color = obj.layersStructs(k).params.leakage * graph.Nodes(obj.layersStructs(k).id).Color;
                    case 'layer'
                        graph.Nodes(k).shape = 'box';
                        graph.Nodes(k).Size = round(80 * [obj.layersStructs(k).numNodes; 0.25*obj.layersStructs(k).numNodes]/obj.layersStructs(k).numNodes) + 20;
                        graph.Nodes(k).Label = [num2str(obj.layersStructs(k).id), ':',obj.layersStructs(k).layerType(1:3), '(',num2str(obj.layersStructs(k).numNodes),')'];
                        graph.Nodes(obj.layersStructs(k).id).Color = obj.layersStructs(k).params.leakage * graph.Nodes(obj.layersStructs(k).id).Color;
                    case 'bias'
                        graph.Nodes(k).shape = 'box';
                        graph.Nodes(k).Size = [15; 15];
                        graph.Nodes(k).Label = num2str(obj.layersStructs(k).id);
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
            %ind = obj.IDtoInd(obj.getIdByName('output'));
            ind = obj.IDtoInd(obj.outputIDs); 
            y = obj.states(obj.blocksInds(ind, 1):obj.blocksInds(ind, 2));
        end
        %-------------------------------------------------------------------------------------
        function setInput(obj, u)
            %ind = obj.IDtoInd(obj.getIdByName('input')); % Here is a bottle neck 
            %obj.setStates(obj.inputIDs, u);
           for ind = obj.IDtoInd(obj.inputIDs)
               inds = obj.blocksInds(ind, 1):obj.blocksInds(ind, 2);
               obj.states(inds) = u(inds - 1); %  - 1 because bias should always 1;
           end     
        end
        %-------------------------------------------------------------------------------------
        function y = predict(obj, input)
            dimOutput = obj.layersStructs(obj.IDtoInd(obj.outputIDs)).numNodes;
            y = zeros(dimOutput, numel(input));
            for k = 1 : numel(input)
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
            inds1 = obj.layersStructs(obj.IDtoInd(id1)).inds;
            inds2 = obj.layersStructs(obj.IDtoInd(id2)).inds;
            W = obj.Weights(inds1, inds2); 
        end
        %-------------------------------------------------------------------------------------
        function x = activateLayer(obj, x, ind)
                x = obj.layersStructs(ind).actFunc(x); %this part a bit slows down
        end
        %-------------------------------------------------------------------------------------
        function setWeights(obj, id1, id2, W)
            inds1 = obj.layersStructs(obj.IDtoInd(id1)).inds;
            inds2 = obj.layersStructs(obj.IDtoInd(id2)).inds;
            obj.Weights(inds1, inds2) = W;
        end
        %-------------------------------------------------------------------------------------
        function setWeightsForSelectedWeights(obj, IDs, ID, W)
            skippedIDs = setdiff(obj.indToID, IDs); % obj.indToID is set of all IDs
            for k = 1:numel(IDs)
                numSkipped = sum(skippedIDs < IDs(k)); % number of skipped layers
                skipInds = obj.blocksInds(skippedIDs(1:numSkipped), 2)...
                        - (obj.blocksInds(skippedIDs(1:numSkipped), 1) - 1); % number of states in every skipped layer 
                % remove the number of states corresponding to the skipped layers
                indsForSkipped = obj.getInds(IDs(k)) - sum(skipInds);
                obj.setWeights(IDs(k), ID, W(indsForSkipped, :));
            end
        end
        %-------------------------------------------------------------------------------------
        function forwardLayer(obj, id)
            layerInd = obj.IDtoInd(id);
            inds = obj.layersStructs(layerInd).inds;
%             obj.states(inds) = obj.leakage(inds) .* obj.layersStructs(obj.IDtoInd(id)).actFunc(obj.Weights(inds, :) * obj.states)...
%                                         + (1 - obj.leakage(inds)) .* obj.states(inds);
            
            leakage = obj.layersStructs(layerInd).params.leakage;                       
            obj.states(inds) =  leakage.* obj.layersStructs(layerInd).actFunc(obj.states * obj.Weights(:, inds))...
                                        + (1 - leakage) .* obj.states(inds); 
        end
        %-------------------------------------------------------------------------------------
        function forward(obj, u) % Calculates next step of the system
            obj.setInput(u);
            
            for k = 1:numel(obj.propogationOrder)
                  obj.forwardLayer(obj.propogationOrder(k));
            end
            %x = obj.states; %remove this, to speed up
        end
     %-------------------------------------------------------------------------------------
       function idList = getIdByName(obj, name)
            idList = [];
            for k = 1:numel(obj.IDtoInd)
                if(strcmp(obj.layersStructs(k).layerType, name))
                    idList = [idList, obj.layersStructs(k).id];
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
%                 indLayer = obj.IDtoInd(id);
%                 stateInds = obj.blocksInds(indLayer,1):obj.blocksInds(indLayer,2);
                stateInds = obj.layersStructs(obj.IDtoInd(id)).inds;
       end
       %-------------------------------------------------------------------------------------
       function states = getStates(obj, IDs)
            states = zeros(1, sum(obj.blocksInds(obj.IDtoInd(IDs), 2) - obj.blocksInds(obj.IDtoInd(IDs), 1) + 1));
            curInd = 0;
            for ind = obj.IDtoInd(IDs)
                numNodes = obj.blocksInds(ind, 2) - obj.blocksInds(ind, 1) + 1;
                states(curInd+1:curInd + numNodes) = obj.states(obj.blocksInds(ind, 1):obj.blocksInds(ind, 2));
                curInd = curInd + numNodes;
            end           
       end
       %-------------------------------------------------------------------------------------
       function setStates(obj, ID, states)
           obj.states(obj.layersStructs(obj.IDtoInd(ID)).inds) = states;
           obj.layersCells
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
                numStates = numStates +  obj.layersStructs(obj.IDtoInd(IDs(k))).numNodes;
            end    
       end
       %-------------------------------------------------------------------------------------
       function propogationRoute = buildPropogationRoute(obj, conMat, inputID, outputID)
            conMat = ceil(conMat);
            startInd = obj.IDtoInd(inputID);
            outputInd = obj.IDtoInd(outputID);
            nodeToBeVisited = startInd; % <<<<<<<<< 
            propogationRoute = [];
            while(~isempty(nodeToBeVisited))
                curInd = nodeToBeVisited(1);
                nodeToBeVisited(1) = [];
                if(sum(ismember(propogationRoute, curInd)) ~= 0)
                    continue;
                end
                candidates = find(conMat(curInd,:));
                dist = [];
                for k = 1:numel(candidates)
                    tempConMat = conMat;
                    tempConMat(curInd, candidates(k)) = 0;
                    % !!! instead of outputNode, search for the closest common (accessable) node
                    dist(k) = graphshortestpath(tempConMat, 2, outputInd);
                end
                
                [~, sind] = sort(dist);
                % rearrange the nodes to visit first
                nodeToBeVisited = [nodeToBeVisited candidates(sind)];
                propogationRoute = [propogationRoute, curInd];
            end
            propogationRoute = obj.indToID(propogationRoute(numel(startInd)+1:end)); % skip input node
       end    
    end    
end