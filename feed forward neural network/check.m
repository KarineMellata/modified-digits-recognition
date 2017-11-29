function [correct, incorrect, final] = check(X, W1_2, W2_3, y_actual, classes, output)
        a = size(output);
        final = zeros(a(2), 1);
        incorrect = 0;
        correct = 0;
        for i = (1:a(2))
            [val, idx] = max(output(:,i));
            [valy, idy] = max(y_actual(i,:));
            final(i) = classes(idx);
            if (idx == idy)
                correct = correct + 1;
            else
                incorrect = incorrect + 1;
            end
        end
