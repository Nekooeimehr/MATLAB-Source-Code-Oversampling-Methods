function labels = Orig_agg_cluster(data, CThresh)

N = size(data,2);

% Clusters is a cell array of vectors.  Each vector contains the
% indicies of the points belonging to that cluster.
% Initially, each point is in it's own cluster.
clusters = cell(N,1);
for cc = 1:length(clusters)
    clusters{cc} = [cc];
end

% the distance between each pair of points
% point_dist = point_distance(data);
D = pdist(data,'euclidean');
point_dist = squareform(D);
point_dist2 = point_dist; 
for i=1:N
    point_dist2(i,i) = 100;
end
thresh = mean(median(point_dist2)).* CThresh;

Z = linkage(D,'complete');
labels = cluster(Z,'cutoff',thresh, 'criterion', 'distance');

function d = point_distance(X)
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
    d = min(d(:))*MaxM^0;
else if version == 2
        d = mean(d(:))*MaxM^0;
    else
        d = max(d(:))*MaxM^0;
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



