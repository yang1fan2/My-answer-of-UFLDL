function [cost,activeLayer] = FWsoftmax(X,y)
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
