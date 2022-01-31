
%% This research is made available to the research community.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% If you are using this code please cite the following paper:                                              %
% Muhammad, U., Hoque, M.Z., Oussalah, M., Keskinarkaus, A., Seppänen, T. and Sarder, P., 2022.             % 
% SAM: Self-augmentation mechanism for COVID-19 detection using chest X-ray images. Knowledge-Based Systems % 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% RICA AUGMENTATION
% Input for RICA augmentation
%%
% Features extraction based on the last average pooling layer of ResNet for the training set 
trainingSet.ReadFcn = @(filename)readAndPreprocessImage(filename);
featureLayer =  'avg_pool';
trainingFeatures = activations(net1, trainingSet, featureLayer, ...
    'MiniBatchSize', 32, 'OutputAs', 'columns');

% Features extraction based on the last average pooling layer of ResNet for the testing set 
% Extract testing set features using the CNN
testing.ReadFcn = @(filename)readAndPreprocessImage(filename);
testingFeatures = activations(net1,testing, featureLayer, ...
    'MiniBatchSize', 32, 'OutputAs', 'columns');
%% RICA 
% Augmentation 1
 q = 500;
 Mdl = rica(trainingFeatures',q,'IterationLimit',80);
 train1 = transform(Mdl,trainingFeatures');
 test1 = transform(Mdl,testingFeatures');
% Augmentation 2
 q = 500;
 Mdl = rica(trainingFeatures',q,'IterationLimit',120);
 train2 = transform(Mdl,trainingFeatures');
 test2 = transform(Mdl,testingFeatures');
% Augmentation 3
 q = 500;
 Mdl = rica(trainingFeatures',q,'IterationLimit',100);
 train3 = transform(Mdl,trainingFeatures');
 test3 = transform(Mdl,testingFeatures');
