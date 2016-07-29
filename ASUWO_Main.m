clear
clc
close all

% Loading the example dataset
load fisheriris
X = meas;
%Y = [ones(100,1); -1 * ones(50,1)];
Y = [ones(50,1); -1 * ones(50,1);ones(50,1)];

[N D] = size(X);
% Standardize the feature sapce
for i = 1:D
    X_scaled(:,i) = 2*((X(:,i) - min(X(:,i))) / ( max(X(:,i)) - min(X(:,i)) ))-1;
end
X_scaled = X_scaled + normrnd(0,0.01,size(X_scaled));

NumberFolds = 3;
NumIteration = 2;

SR_RG = 1;
stepSize = 1;

division = round(N/NumberFolds);

%% Buiding the models
for ite = 1:NumIteration
    C = cvpartition(Y,'k',NumberFolds);
    for num = 1:NumberFolds;
        trainData = X_scaled(training(C,num),:);
        trainLabel = Y(training(C,num),:);
        testData = X_scaled(test(C,num),:);
        testLabel = Y(test(C,num),:);
        %% Oversampling using SMOTE
        display ('SMOTE:')
        [trainDataSMOTE, trainLabelSMOTE] = SMOTE(trainData,trainLabel);
        %% Oversampling using Borderline SMOTE
        display ('Borderline SMOTE:')
        NNC = 5;
        [borderMin_BorSMOTE, trainDatanewBorSMOTE, trainLabelnewBorSMOTE] = BorSMOTE(trainData,trainLabel,NNC);
        %% Oversampling using Safe-level SMOTE
        display ('Safe-level SMOTE:')
        NNC = 5;
        [trainDatanewSafeSMOTE, trainLabelnewSafeSMOTE] = Safe_Level_SMOTE(trainData,trainLabel,NNC);
        %% Oversampling using ASUWO
        display ('ASUWO:')
        CThresh = 1;
        K = 3;
        NN = 5;
        NS = 5;
        [trainDatanewASUWO, trainLabelnewASUWO] = ASUWO(trainData,trainLabel, CThresh , K, NN, NS);
    end
    perm = [];
end