function [data_product] = cartprod(data)
% CARTPROD create cartesian product of data items

columns = length(data);
rows = zeros(1, columns);
for i=1:columns
    rows(i) = length(data{i});
end

column_idx = ones(1, columns);
cartprod_count = 0;
max_cartprod_count = prod(rows);
cartprod_idx = ones(max_cartprod_count, columns);
while cartprod_count < max_cartprod_count
    cartprod_count = cartprod_count + 1;
    cartprod_idx(cartprod_count, :) = column_idx;
    column_idx(1) = column_idx(1) + 1;
    for i=1:columns-1
        if column_idx(i) <= rows(i)
            break
        end
        column_idx(i) = 1;
        column_idx(i+1) = column_idx(i+1) + 1;
    end
    if column_idx(end) > rows(end)
        break
    end    
end

data_product = zeros(size(cartprod_idx));
[height, width] = size(data_product);
for y = 1:height    
    idx = cartprod_idx(y,:);    
    for x=1:width
        p = data{x};
        data_product(y,x) = p(idx(x));
    end
end


end
