function [Kmin2, rand_matrix, Final_Ov] = Num_OV_Finder(IDX_min, Majority_features, Minority_features, m_each_min, Kmin, folds, Out_Th)

pow = 0.2;
Maj_size = size(Majority_features,1);
Min_size = size(Minority_features,1);
% Randomely permute the memebrs in each minority cluster
rand_matrix = [];
for i=1:Kmin
    perm = [];
    buff_min_ind = find(IDX_min == i);
    des_min_sam = size(find(IDX_min == i),1);
    perm = randsample(buff_min_ind,des_min_sam);
    padadd(rand_matrix,perm)
end
rand_matrix(:, m_each_min <= Out_Th) = [];
m_each_min(m_each_min <= Out_Th) = [];
Kmin2 = size(rand_matrix,2);
LessFoldsIn = find(m_each_min<folds);
if size(LessFoldsIn,1)>=1
    for fk = 1:size(LessFoldsIn,1)
        temp1 = rand_matrix(~isnan(rand_matrix(1:end,LessFoldsIn(fk))),LessFoldsIn(fk));
        Added = randsample(temp1,folds-size(temp1,1),true);
        rand_matrix ((size(temp1,1)+1):folds,LessFoldsIn(fk)) = Added;
    end
end

% Split each Minority cluster and put some portion of each in the fold matrix
buffer = [] ;
folds_matrix = [];
for i = 1:folds-1
    for j=1:Kmin2 
        temp = rand_matrix(~isnan(rand_matrix(1:end,j)),j);
        division = floor(size(temp,1)/folds);
        buffer = [buffer; temp(((i-1)*division+1):i*division,1)];
    end
    padadd(folds_matrix,buffer)
    buffer = [];
end

for j=1:Kmin2
    temp = rand_matrix(~isnan(rand_matrix(1:end,j)),j);
    division = floor(size(temp,1)/folds);
    buffer = [buffer; temp(((folds-1)*division+1):end,1)];
end
padadd(folds_matrix,buffer)

% Finding the number of misclassified instances 
errorCluster_min = zeros(1,Kmin2);
C = nchoosek(1:folds,folds-1);
% for ite = 1:folds
ite = 1;
    A_min = folds_matrix(:,C(ite,:));
    Min_Feat_Train = Minority_features(A_min(~isnan(A_min)),:);
    B_min = folds_matrix(:,~ismember(1:folds,C(ite,:)));
    Min_Feat_Valid = Minority_features(B_min(~isnan(B_min)),:);    
    % Train the SVM
    Feat_Train_whole = [Min_Feat_Train; Majority_features];
    trainLabel_whole = [-1*ones(size(Min_Feat_Train,1),1);ones(Maj_size,1)];
    [trainDatanew, trainLabelnew] = SMOTE(Feat_Train_whole, trainLabel_whole);
    %model = svmtrain(trainLabelnew, trainDatanew, Options);
    model = fitcdiscr(trainDatanew, trainLabelnew);
    
    % Use the LDA/SVM model to classify the data
    predict_label_SMOTE = predict(model, Min_Feat_Valid);
    % predict_label_SMOTE = svmpredict(testLabel, Min_Feat_Valid, model, '-q'); % run the SVM model on the test data
    misclassified = B_min(predict_label_SMOTE == 1);
    errorCluster_min = sum(ismember(rand_matrix,misclassified)) + errorCluster_min;
%end

NeedOv = Maj_size - Min_size;
% Kmin_real = size(m_each_min_real,1);
Pow_m_each = m_each_min .^ pow;
Ratio_Size = (1./Pow_m_each)/sum(1./Pow_m_each,1);

ratio_min = errorCluster_min./sum(~isnan(rand_matrix));
ratio_min(ratio_min <= 0.1)= 0.1;
ratio_min2 = ratio_min/sum(ratio_min);

% ratio_min2(ratio_min2 <= 0.1)= 0.1;
New_Ratio = ratio_min2 .* Ratio_Size'; 
ratio_min_scaled = New_Ratio/sum(New_Ratio)
Final_Ov = floor(NeedOv * ratio_min_scaled)

end