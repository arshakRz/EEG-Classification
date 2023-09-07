function [fisher] = fisher_score(feat, label)
    fisher = zeros(1,size(feat,2));
    for i = 1:size(feat,2)
        v = feat(:,i);
        u0 = mean(v);
        u1 = mean(v(label==1));
        u2 = mean(v(label==2));
        var1 = var(v(label==1));
        var2 = var(v(label==2));
        fisher(i) = (abs(u0-u1)^2 + abs(u0-u2)^2)/(var1+var2);
    end
end