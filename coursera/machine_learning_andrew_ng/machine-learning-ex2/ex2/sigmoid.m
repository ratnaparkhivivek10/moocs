function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).

g = 1./(1+exp(-z));
% "./ vs /" - The regular arithmetic operators will become element-by-element operators if you place a dot in front of them.
% http://www.malinc.se/math/octave/rowscolumnsen.php


% =============================================================

end
