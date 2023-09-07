function [cost] = fisher_score_nd(feat)
    load top30_label.mat
    top30 = top30_label(:,1:30);
    label = top30_label(:,31)';
    v = top30(:,feat==1);
    v1 = v(label==1,:);
    v2 = v(label==2,:);
    s1 = zeros(size(v,2));
    s2 = zeros(size(v,2));
    for i = 1:60
        s1 = s1 + (v1(i,:)-mean(v1,1))' * (v1(i,:)-mean(v1,1));
        s2 = s2 + (v2(i,:)-mean(v2,1))' * (v2(i,:)-mean(v2,1));
    end
    sw = 1/60*(s1+s2);
    sb = (mean(v1,1)-mean(v,1))'*(mean(v1,1)-mean(v,1)) ...
        +(mean(v2,1)-mean(v,1))'*(mean(v2,1)-mean(v,1));
    %cost = -trace(sb)/trace(sw);
    %cost = -det(sb)/det(sw);
    cost = 1/trace(sb .* sw.^-1);
end