function [final_features ,final_mark] = ClusterSMOTE(original_features, original_mark, Ncluster)

ind = find(original_mark == -1);
Min_instances = original_features(ind,:);
min_clusters = kmeans(Min_instances,Ncluster);

KNN = 6;
final_features = original_features;

Num_Ov = ceil(max(size(find(original_mark == -1),1) - size(find(original_mark == 1),1),size(find(original_mark == 1),1) - size(find(original_mark == -1),1)));
j2 = 1;


while j2 <= Num_Ov
    %find nearest K samples from S2(i,:)
    [S2 idx]= datasample(Min_instances,1);
    Min_Cluster = find(min_clusters == min_clusters(idx));
    Min_cand = Min_instances(Min_Cluster,:);
    Limit = size(Min_cand,1);
    Condidates = nearestneighbour(S2', Min_cand', 'NumberOfNeighbours', min(KNN,Limit));
    Condidates(:,1) = [] ;
    if size(Condidates,2)>= 1
        rn=ceil(rand(1)*(size(Condidates,2)));
        Sel_index = Condidates(:,rn);
        g = Min_instances(Sel_index,:);
        alpha = rand(1);
        snew = S2(1,:) + alpha.*(g-S2(1,:));
        final_features = [final_features;snew];
        j2=j2+1;
    end
end

mark = -1 * ones(Num_Ov,1);
final_mark = [original_mark; mark];