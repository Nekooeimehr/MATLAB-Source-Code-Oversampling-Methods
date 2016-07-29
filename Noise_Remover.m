function [ClearData, ClearLabel] = Noise_Remover(WholeDataInst, WholeDataLable, KNN)  

Ins_neighbors = knnsearch(WholeDataInst, WholeDataInst, 'k', KNN);
Safe_Level = zeros(1,size(WholeDataInst,1));

for i = 1:size(WholeDataInst,1)
    for j = 2:KNN
        if(WholeDataLable(Ins_neighbors(i,j),1) == WholeDataLable(i,1))
            Safe_Level(1,i) = Safe_Level(1,i) + 1;
        end
    end
end

ToRemove = find(Safe_Level == 0);
ClearData = WholeDataInst;
ClearData(ToRemove,:) = [];
ClearLabel = WholeDataLable;
ClearLabel(ToRemove,:) = [];

end