function output = softmax(h)

output = exp(h')./(sum(exp(h')));
