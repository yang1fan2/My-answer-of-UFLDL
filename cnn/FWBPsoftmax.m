function [cost,gradient,activeLayer] = FWBPsoftmax(X,y)
  % X : C * M
  %gradient : C*M
  m=size(X,2);
  n=size(X,1);

  
  % initialize objective value and gradient.
  cost = 0;
  gradient = zeros(size(X));
  h = exp(X);
  activeLayer = bsxfun(@rdivide,h,sum(h,1));
  if (size(y) == 0)
    return
  end
  P = log(activeLayer);
  I = sub2ind(size(P), y', 1:m);
  values = P(I);
  cost = -sum(values);
  P = activeLayer;
  P(I) = P(I)-1;
  gradient = P;
  
  %
  % TODO:  Compute the softmax objective function and gradient using vectorized code.
  %        Store the objective function value in 'f', and the gradient in 'g'.
  %        Before returning g, make sure you form it back into a vector with g=g(:);
  %
%%% YOUR CODE HERE %%%
  % make gradient a vector for minFunc

