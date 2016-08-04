[obj.W_in{1} * [1; u] ;  obj.W(1:10,1:10) * ones(10,1);  obj.W_fb{1} * obj.Y_last]
states = [1; u; ones(10,1); obj.Y_last]
[obj.W_in{1};   obj.W(1:10,1:10) ;  obj.W_fb{1}]

D = blkdiag(obj.W_in{1}, obj.W(1:10,1:10), obj.W_fb{1})

D * states

[obj.W_in{1} full(obj.W(1:10,1:10)) obj.W_fb{1};  [Wout, [0;0]]]