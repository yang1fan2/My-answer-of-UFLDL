function [ cost, grad, pred_prob] = supervised_dnn_cost( theta, ei, data, labels, pred_only)
%SPNETCOSTSLAVE Slave cost function for simple phone net
%   Does all the work of cost / gradient computation
%   Returns cost broken into cross-entropy, weight norm, and prox reg
%        components (ceCost, wCost, pCost)

%% default values
po = false;
if exist('pred_only','var')
  po = pred_only;
end;

%% reshape into network
stack = params2stack(theta, ei);
numHidden = numel(ei.layer_sizes) - 1;
indexOutput = numHidden + 1;
A = cell(numHidden+1, 1); % cur * M
Z = cell(numHidden+1, 1)

gradStack = cell(numHidden+1, 1);
%% forward prop
%%% YOUR CODE HERE %%%
for I = 1:numHidden+1
	I
	if (I ==1)
		Z{I} = bsxfun(@plus,stack{I}.W * data, stack{I}.b);
		A{I} = sigmoid(Z{I});
	else 
		Z{I} = bsxfun(@plus,stack{I}.W * A{I-1}, stack{I}.b);
		A{I} = sigmoid(Z{I});		
	end
	size(A{I})
end


%% return here if only predictions desired.
if po
  cost = -1; ceCost = -1; wCost = -1; numCorrect = -1;
  grad = [];  
  return;
end;

%% compute cost
%%% YOUR CODE HERE %%%

[ceCost, Error,A{indexOutput}] = FWBPsoftmax(Z{indexOutput},labels);
pred_prob = A{indexOutput}
%% compute gradients using backpropagation
%%% YOUR CODE HERE %%%

for I = numHidden+1:-1:1
	if (I == 1)
		gradStack{I}.W = Error * data;
	else
		gradStack{I}.W = Error * A{I-1}';
	end
	gradStack{I}.b = sum(Error, 2);
	Error = (stack{I}.W' * Error) .* ferval(['BP',ei.activation_fun],Z{I-1},A{I-1});

end

%% compute weight penalty cost and gradient for non-bias terms
%%% YOUR CODE HERE %%%
wCost = 0;
for I = 1:numHidden+1
	wCost +=  ei.lambda * sum(stack{I}.W(:).^2)/2;
end

cost = wCost + ceCost;

for l = 1:numHidden+1
	gradStack{I}.W += stack{I}.W;
end


%% reshape gradients into vector
[grad] = stack2params(gradStack);
end



