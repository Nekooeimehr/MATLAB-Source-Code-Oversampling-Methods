function [final_features ,final_mark] = Safe_Level_SMOTE(original_features, original_mark, KNN)

ind = find(original_mark == -1);
Min_ins = original_features(ind,:);
KNN = KNN + 1;
final_features = original_features;
Limit = size(Min_ins,1);

Num_Ov = ceil(max(size(find(original_mark == -1),1) - size(find(original_mark == 1),1),size(find(original_mark == 1),1) - size(find(original_mark == -1),1)));
j2 = 1;

Safe_Level = safe_level_Finder(Min_ins, original_features, original_mark, KNN);

while j2 <= Num_Ov
    %find nearest K samples from S2(i,:)
    [FirstCand idx] = datasample(Min_ins,1);
    Safe_Level_cand1 = Safe_Level(idx);
    Condidates = nearestneighbour(FirstCand', Min_ins', 'NumberOfNeighbours', min(KNN,Limit));
    Condidates(:,1) = [] ;
    rn=ceil(rand(1)*(size(Condidates,2)));
    Sel_index = Condidates(:,rn);
    SecondCand = Min_ins(Sel_index,:);
    Safe_Level_cand2 = Safe_Level(Sel_index);
    
    if Safe_Level_cand2 ~= 0
    Safe_level_ratio = Safe_Level_cand1/Safe_Level_cand2;
    else
        Safe_level_ratio = inf;
    end
    
    if (Safe_level_ratio == inf && Safe_Level_cand1 == 0)
    else
        if (Safe_level_ratio == inf && Safe_Level_cand1 ~= 0)
            gap = 0;
        else if Safe_level_ratio == 1
                gap = rand(1);
            else if Safe_level_ratio > 1
                    gap = rand(1)*(1/Safe_level_ratio);
                else if Safe_level_ratio < 1
                        gap = rand(1) * Safe_level_ratio + 1 - Safe_level_ratio;
                    end
                end
            end
        end
    snew = FirstCand(1,:) + gap.*(SecondCand - FirstCand(1,:));
    final_features = [final_features;snew];
    j2=j2+1;    
    end    
end

mark = -1 * ones(Num_Ov,1);
final_mark = [original_mark; mark];
end

function Safe_Level = safe_level_Finder(Minority_features, WholeDataInst, WholeDataLable, KNN)  

Ins_neighbors = nearestneighbour(Minority_features', WholeDataInst', 'NumberOfNeighbours', KNN);
Safe_Level = zeros(1,size(Minority_features,1));

for i = 1:size(Minority_features,1)
    for j = 2:KNN
        if(WholeDataLable(Ins_neighbors(j,i),1)== -1)
            Safe_Level(1,i) = Safe_Level(1,i) + 1;
        end
    end
end

end