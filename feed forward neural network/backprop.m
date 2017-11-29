function [W1_2, W2_3] = backprop(X, y, W1_2, W2_3, alpha, lambda)

    h1 = sigmoid(X * W1_2);
    outputNode = softmax(h1 * W2_3);
    softmax_grad = (y - outputNode')' * h1;
    h1_grad = ((y - outputNode')*W2_3)'*h1*(1-h1)'*X;
    h1_grad = h1_grad';
    W2_3 = W2_3 - alpha*softmax_grad + 2*lambda*W2_3;
    
    W1_2 = W1_2 - alpha*h1_grad - 2*lambda*W1_2;
end