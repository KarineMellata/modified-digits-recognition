
function sigmoid = sig(h)

a = exp(h);
sigmoid = a ./ (exp(h) + 1);
end
