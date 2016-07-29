function [final_features, final_mark] = ASUWO(original_features, original_mark, CThresh , K, NN, NS)

%Inputs:
    % original_features: The features of original dataset needed to be oversampled. 
    % original_mark: The label of original dataset needed to be oversampled.
	% CThresh: Coefficient to tune the threshold for clustering.
	% NN: Number of nearest neighbors to be found for each minority instance to determine the weights.
	% NS: Number of nearest neighbors used to identify noisy instances. 
	% K: Number of folds in the K-fold Cross Validation.

%Outputs:
    % final_features: The features of dataset after being oversampled.
    % final_mark: The label of dataset after being oversampled.
% Copyright 2015 Iman Nekooeimehr. This code may be freely used and
% distributed, so long as it maintains this copyright line.
    
%Removing noisy instances for both minority and majority class:
[Clean_orig_inst, Clean_orig_mark] = Noise_Remover(original_features, original_mark, NS); 
    
NNC = 5;
Out_Th = 2;

% Separating Minority and Majority instances
MinorityIndex = find(Clean_orig_mark == -1);
MajorityIndex = find(Clean_orig_mark == 1);
Majority_features = Clean_orig_inst(MajorityIndex,:);
Minority_features = Clean_orig_inst(MinorityIndex,:);
Maj_size = size(Majority_features,1);

%% Clustering the minority instances by considering majority instances:
[IDX_min] = Mod_AggCluster(Majority_features, Minority_features ,CThresh);
Kmin = size(unique(IDX_min),1);
m_each_min = histc(IDX_min,1:Kmin);

%% Finding cluster sizes for minority class using K fold cross validation
[Kmin2, rand_matrix, num_cluster_min] = Num_OV_Finder(IDX_min, Majority_features, Minority_features, m_each_min, Kmin, K, Out_Th);

final_features = Minority_features;

%% find selection probability and oversample within minority clusters
[p,q]=size(Majority_features);

for i=1:Kmin2
    minority_clustered = rand_matrix(~isnan(rand_matrix(:,i)),i);
    Minority_clustered_features = Minority_features(minority_clustered,:);
    [m,n]=size(Minority_clustered_features);
    dist_vec = [];
    for i2=1:m
        %find nearest K1 borderline majority sets
        dist = zeros(p,1);
        for j=1:p
            x = sum((Majority_features(j,:) - Minority_clustered_features(i2,:)).^2);
            dist(j,1) = x;
        end
        distm = sort (dist);
        dist_vec = [dist_vec distm(1:NN)];
    end
    thresh = quantile(dist_vec(1,:),0.5);
    dist_vec(dist_vec > thresh) = thresh;
    dist_vec = dist_vec./n;
    dist_rec = (1./dist_vec).^1;
    mean_dis = mean(dist_rec,1);
    totw = sum(mean_dis);
    P = mean_dis ./ totw;
    %end of our selection probability algorithm
    
    j2 = 1;
    while j2 <= (num_cluster_min(1,i))
        %find nearest K samples from S2(i,:)
        S2=sample(Minority_clustered_features,P',1);
        Condidates = nearestneighbour(S2', Minority_clustered_features', 'NumberOfNeighbours', min(NNC,m));
        Condidates(:,1) = [] ;
        rn=ceil(rand(1)*(size(Condidates,2)));
        Sel_index = Condidates(:,rn);
        g = Minority_clustered_features(Sel_index,:);
        alpha=rand(1) ;
        snew = S2(1,:) + alpha.*(g-S2(1,:));
        final_features = [final_features;snew];
        j2=j2+1;
    end    
end

r = size(final_features,1);
MinMark = -1 * ones(r,1);
MaxMark = ones(Maj_size,1);
final_mark = [MinMark; MaxMark];
final_features = [final_features; Majority_features];

