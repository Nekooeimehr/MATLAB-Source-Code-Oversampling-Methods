function [min_clusters] = Mod_AggCluster(Majority_features, Minority_features ,CThresh)

% This code is a modification of the source code for Hierachical Clustering
% implemented by David Ross
% The source code for the original Hierachical Clustering can be found in: 
% http://www.cs.toronto.edu/~dross/code/

SizeMin = size(Minority_features,1);
min_clusters = (1:SizeMin)';

%% Clustering the majority class using Hierachical Clustering
maj_clusters = Orig_agg_cluster(Majority_features, CThresh);

% Kmaj = size(unique(maj_clusters),1);
% m_each_maj = histc(maj_clusters,1:Kmaj);

Whole_data_min = [Minority_features; Majority_features];
D = pdist(Whole_data_min,'euclidean');
point_dist_min = squareform(D);

%% Clustering the Minority instances using majority clusters
min_clusters = inside_AggCluster(Minority_features', min_clusters, maj_clusters, point_dist_min, CThresh);

function labels  = inside_AggCluster(data, same_clusters, other_clusters, point_dist_whole, CThresh)
Num_Reject = 0; 
N = size(data,2);
Exist_Clus = unique(same_clusters);
M = size(Exist_Clus ,1);

% the distance between each pair of points
point_dist = point_dist_whole(1:N,1:N);
point_dist2 = point_dist; 
for i=1:N
    point_dist2(i,i) = 100;
end

% Measuring the threshold
thresh = mean(median(point_dist2)).* CThresh;

% Clusters is a cell array of vectors.  Each vector contains the
% indicies of the points belonging to that cluster.
% Initially, each point is in it's own cluster.
clusters = cell(M,1);
for cc = 1:M
    clusters{cc} = find(same_clusters == Exist_Clus(cc))';
end

% until the termination condition is met
mm = 0;                                                    
while mm < thresh
    
    % compute the distances between all pairs of clusters
    cluster_dist = inf*ones(length(clusters));
    for c1 = 1:length(clusters)
        for c2 = (c1+1):length(clusters)
            cluster_dist(c1,c2) = cluster_distance(clusters{c1}, clusters{c2}, point_dist, 3);
        end
    end
    
    % merge the two nearest clusters
    [mm ii] = min(cluster_dist(:));
    [ii(1) ii(2)] = ind2sub(size(cluster_dist), ii(1));
    
    if mm > thresh || length(clusters) < 3,
        break
    end
    % find the distance of nearest clusters to other class clusters:
    Unique_Other = unique(other_clusters);
    num_clus = size(Unique_Other,1);
    
    for k = 1:num_clus
        MN2other(k) = cluster_distance_maj(clusters{ii(1)}, N + find(other_clusters == Unique_Other(k)), point_dist_whole, 3);
    end
    flag = 1;
    Distr = histc(other_clusters,1:max(other_clusters));
    Distr(Distr == 0) = [] ;
    near_other_ind = find(MN2other < mm & Distr' > 3);
    for t = 1:length(near_other_ind)
        check_dis = cluster_distance_maj(clusters{ii(2)}, N + find(other_clusters == Unique_Other(near_other_ind(t))) , point_dist_whole, 3);
        if check_dis <mm
            flag = 0;
            Num_Reject = Num_Reject + 1;
            A = clusters{ii(1)};
            B = clusters{ii(2)};
            point_dist (A(1,1),B(1,1)) = inf;
            point_dist (B(1,1),A(1,1)) = inf;
        end
    end
    % Place the if condition if there exist a majority cluster between them or not
    if flag == 1;
        clusters = merge_clusters(clusters, ii);
    end
end

% assign labels to the points, based on their cluster membership
Num_Reject
labels = zeros(N,1);
for cc = 1:length(clusters)
    labels(clusters{cc}) = cc;
end



%//////////////////////////////////////////////////////////
% d = point_distance(X)
%    Computes the pairwise distances between columns of X.
%----------------------------------------------------------
function d = Point_Distance(X)
N = size(X,2);
d = sum(X.^2,1);
d = ones(N,1)*d + d'*ones(1,N) - 2*X'*X;



%//////////////////////////////////////////////////////////
% d = cluster_distance(c1,c2,point_dist,linkage)
%    Computes the pairwise distances between clusters c1
%    and c2, using the point distance info in point_dist.
%----------------------------------------------------------
function d = cluster_distance(c1,c2,point_dist,version)

M1 = length(c1);
M2 = length(c2);
MaxM = max([M1,M2]);
d = point_dist(c1,c2);
if version == 1
    d = min(d(:))*MaxM^0.04;
else if version == 2
        d = mean(d(:))*MaxM^0.04;
    else
        d = max(d(:))*MaxM^0.04;
    end
end

function d = cluster_distance_maj(c1,c2,point_dist,version)
d = point_dist(c1,c2);
if version == 1
    d = min(d(:));
else if version == 2
        d = mean(d(:));
    else
        d = max(d(:));
    end
end
%//////////////////////////////////////////////////////////
% clusters = merge_clusters(clusters, indicies)
%   Merge the clusters indicated by the entries indicies(1)
%   and indicies(2) of cell array 'clusters'.
%----------------------------------------------------------
function clusters = merge_clusters(clusters, indicies)
clusters{indicies(1)} = [clusters{indicies(1)} clusters{indicies(2)}];
clusters(indicies(2)) = [];

