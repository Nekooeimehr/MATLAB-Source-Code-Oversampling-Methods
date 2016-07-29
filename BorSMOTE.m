function [border_min, final_features, final_mark] = BorSMOTE(original_features, original_mark, NNC)

%NNC = NNC + 1;

Minority_index = find(original_mark == -1);
Minority_features = original_features(Minority_index,:);

% Finding the 5 positive nearest neighbours of all the positive blobs
Minority_neighbors = nearestneighbour(Minority_features', original_features', 'NumberOfNeighbours', NNC);

num_min_neighbor = zeros(1,length(Minority_index));
for i=1:length(Minority_index)
    for j = 2:NNC
        if(original_mark(Minority_neighbors(j,i),1)== 1) 
            num_min_neighbor(1,i) = num_min_neighbor(1,i)+1;
        end
    end
end

border_min = Minority_index(find(num_min_neighbor > (NNC-1)/2),1); 
while size( border_min,1) < 4
    NNC = NNC - 1;
    border_min = Minority_index(find(num_min_neighbor > (NNC-1)/2),1);
end
Border_min_features = original_features(border_min,:);
NNC = 5;
Num_Ov = ceil(max(size(find(original_mark == -1),1) - size(find(original_mark == 1),1),size(find(original_mark == 1),1) - size(find(original_mark == -1),1)));
j2 = 1;
Limit = size(Border_min_features,1);

if Limit > 3
    final_features = original_features;
    while j2 <= Num_Ov
        %find nearest K samples from S2(i,:)
        S2 = datasample(Border_min_features,1);
        Condidates = nearestneighbour(S2', Minority_features', 'NumberOfNeighbours', min(NNC-1,Limit));
        Condidates(:,1) = [] ;
        rn = ceil(rand(1)*(size(Condidates,2)));
        Sel_index = Condidates(:,rn);
        g = Minority_features(Sel_index,:);
        alpha = rand(1);
        snew = S2(1,:) + alpha.*(g-S2(1,:));
        final_features = [final_features;snew];
        j2=j2+1;
    end 
mark = -1 * ones(Num_Ov,1);
final_mark = [original_mark; mark];
else 
    [final_features ,final_mark] = SMOTE(original_features, original_mark);
end




