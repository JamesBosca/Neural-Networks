function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));
% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

a1 =[ones(m,1) X];

z2=Theta1*a1';
[n,~]= size(z2);

z2=[ones(n,1) z2];

a2=sigmoid(z2);
a2=[ones(1,m+1);a2];
size(a2)
z3=Theta2*a2;

h=sigmoid(z3);

vy=zeros(length(y),num_labels);
for i=1:m
 
vy(i,y(i))=1;
end

y=vy;

%Ver "eye(num_labels)(y,:)" que más optimo que el metodo arriba empleado%

for i=1:m
    for k=1:num_labels
        
        J=J+(1/m)*(-y(i,k)*log(h(k,i+1))-((1-y(i,k))*log(1-h(k,i+1))));

    end    
end

Theta1_reg=0;

for j=1:hidden_layer_size
   for k=1:input_layer_size
       Theta1_reg=Theta1_reg+(lambda/(2*m))*(Theta1(j,k+1))^2;
   end
end

Theta2_reg=0;

for j=1:num_labels
    for k=1:hidden_layer_size
        Theta2_reg=Theta2_reg+(lambda/(2*m))*(Theta2(j,k+1))^2;
    end
end

J=J+(Theta1_reg+Theta2_reg);

fprintf('size h');
size(h)

fprintf('size y');
size(y)

%Ojo delta 3 es a3( h) - y ; "como h (a3) tiene el bias unit hay que restar
%a partir de la segunda columna"%

delta3=zeros(num_labels,m);
for k=1:num_labels
for t=1:m
    
delta3(k,t)=h(k,t+1)-y(t,k);

end
end

fprintf('size delta3');
size(delta3)
%Ojo ?2 equals the product of ?3 and ?2 (ignoring the ?2 bias units)%

Theta2=Theta2(:,2:end);

fprintf('size Theta2');
size(Theta2)

deltaint=(transpose(Theta2)*delta3);

z2=z2(:,2:end);
fprintf('size z2');
size(z2)

delta2=deltaint.*sigmoidGradient(z2);

fprintf('size delta2');
size(delta2)

fprintf('size a2');
size(a2)

Delta2=zeros(num_labels,hidden_layer_size+1);
Delta1=zeros(hidden_layer_size+1,input_layer_size+1);

%a2 tiene añadido la comumna de bias unit así que hay que quitarla para
%multiplicar con delta3%
a2=a2(:,2:end);

Delta2=(delta3*transpose(a2));


fprintf('size delta2');
size(delta2)
Delta1=(delta2*a1);

fprintf('size Delta1');
size(Delta1)

%Theta1_grad y 2 tienen que tener las mismas dimensiones que Theta1 y 2%
Theta1_grad=(1/m)*Delta1;

Theta2_grad=(1/m)*Delta2;

fprintf('size Theta2_grad');
size(Theta2_grad)

%COMPUTAR LA REGULARIZACIÓN%
%La primera columna de los Delta no sufre regularización%


Column1Theta1_grad=Theta1_grad(:,1);

Column1Theta2_grad=Theta2_grad(:,1);
%Lo que sufre regularización%

RegTheta1_grad=Theta1_grad(:,2:end)+(lambda/m)*(Theta1(:,2:end));

%Ojo he añadidola primera columna de Theta2 y va bien pero no lo entiendo
%porque aqui si y en el otro no, VER BIEN%

RegTheta2_grad=Theta2_grad(:,2:end)+(lambda/m)*(Theta2);

Theta1_grad=[Column1Theta1_grad RegTheta1_grad];
Theta2_grad=[Column1Theta2_grad RegTheta2_grad];
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
