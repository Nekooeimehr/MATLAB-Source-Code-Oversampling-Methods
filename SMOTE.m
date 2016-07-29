function [final_features ,final_mark] = SMOTE(original_features, original_mark)

ind = find(original_mark == -1);
P = original_features(ind,:);
KNN = 5;
final_features = original_features;
Limit = size(original_features,2);

Num_Ov = ceil(max(size(find(original_mark == -1),1) - size(find(original_mark == 1),1),size(find(original_mark == 1),1) - size(find(original_mark == -1),1)));
j2 = 1;
while j2 <= Num_Ov
    %find nearest K samples from S2(i,:)
    S2= datasample(P,1);
    Condidates = nearestneighbour(S2', P', 'NumberOfNeighbours', min(KNN,Limit));
    Condidates(:,1) = [] ;
    rn=ceil(rand(1)*(size(Condidates,2)));
    Sel_index = Condidates(:,rn);
    g = P(Sel_index,:);
    alpha = rand(1);
    snew = S2(1,:) + alpha.*(g-S2(1,:));
    final_features = [final_features;snew];
    j2=j2+1;
end

mark = -1 * ones(Num_Ov,1);
final_mark = [original_mark; mark];



