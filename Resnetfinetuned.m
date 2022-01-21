%% This research is made available to the research community.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% If you are using this code please cite the following paper:                                              %
% Muhammad, U., Hoque, M.Z., Oussalah, M., Keskinarkaus, A., Seppänen, T. and Sarder, P., 2022.             % 
% SAM: Self-augmentation mechanism for COVID-19 detection using chest X-ray images. Knowledge-Based Systems % 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all
close all
clc
%%  Input for training set and Testing
rootFolder = fullfile('D:\Covid detection\covid');
categories1 = {'COVID','non-COVID'};
trainingSet1 = imageDatastore(fullfile(rootFolder, categories1), 'IncludeSubfolders',true, ...
 'LabelSource','foldernames');
[trainingSet, testing] = splitEachLabel(trainingSet1,0.8);
trainingLabels = trainingSet.Labels;

%% Model input

net = resnet50;
net.Layers(1);

inputSize = net.Layers(1).InputSize;

if isa(net,'SeriesNetwork') 
  lgraph = layerGraph(net.Layers); 
else
  lgraph = layerGraph(net);
end

%%
% freezing initial weights
[learnableLayer,classLayer] = findLayersToReplace(lgraph);
[learnableLayer,classLayer]
numClasses = numel(categories(trainingSet.Labels));
if isa(learnableLayer,'nnet.cnn.layer.FullyConnectedLayer')
    newLearnableLayer = fullyConnectedLayer(numClasses, ...
        'Name','new_fc', ...
        'WeightLearnRateFactor',10, ...
         'WeightL2Factor',1, ...
            'BiasLearnRateFactor',20, ...
        'BiasL2Factor',0);
    
elseif isa(learnableLayer,'nnet.cnn.layer.Convolution2DLayer')
    newLearnableLayer = convolution2dLayer(1,numClasses, ...
        'Name','new_conv', ...
        'WeightLearnRateFactor',10, ...
            ' BiasLearnRateFactor',20);
end

lgraph = replaceLayer(lgraph,learnableLayer.Name,newLearnableLayer);

newClassLayer = classificationLayer('Name','new_classoutput');
lgraph = replaceLayer(lgraph,classLayer.Name,newClassLayer);

layers = lgraph.Layers;
connections = lgraph.Connections;

layers(1:10) = freezeWeights(layers(1:10));
lgraph = createLgraphUsingConnections(layers,connections);


%%
% CNN Model training 

miniBatchSize = 32;
options = trainingOptions('sgdm', ...
    'ExecutionEnvironment','gpu', ... 
    'MiniBatchSize',miniBatchSize, ...
    'MaxEpochs',5, ...
    'InitialLearnRate',3e-4, ...
    'Shuffle','every-epoch', ...
    'Verbose',true, ...
     'Plots','training-progress');

trainingSet.ReadFcn = @(filename)readAndPreprocessImage(filename);

net1 = trainNetwork(trainingSet,lgraph,options);
%%
% Model testing 
testing.ReadFcn = @(filename)readAndPreprocessImage(filename);

[predictedLabels, scores] = classify(net1 ,testing);

% Get the known labels
testLabels = testing.Labels;

% Tabulate the results using a confusion matrix.
confMat = confusionmat(testLabels', predictedLabels);

% Convert confusion matrix into percentage form
confMat = bsxfun(@rdivide,confMat,sum(confMat,2));
%
YValidation =  testLabels;

CPredicted=predictedLabels;

C = confusionmat(YValidation,CPredicted);

OverallAcc = sum(YValidation==CPredicted)/length(CPredicted);
              
Acc = zeros(2,1);

Sens = zeros(2,1);

Spec  = zeros(2,1);

Prec   = zeros(2,1);

F1Sc  = zeros(2,1);

MCC  = zeros(2,1);

FPR  = zeros(2,1);

ERate  = zeros(2,1);

 for i=1:length(C)
      TP=C(i,i);
      TN = sum(C(:))-sum(C(:,i))-sum(C(i,:))+C(i,i);
      FP=sum(C(:,i))-C(i,i);
      FN = sum(C(i,:))-C(i,i);
      Acc(i)= (TP+TN)/(TP+TN+FP+FN);
      Sens(i) = TP/(TP+FN);
      Spec(i) = TN/(FP+TN);
      Prec(i) = TP/(TP+FP);
      F1Sc(i) = 2*TP/(2*TP+FP+FN);
      S = (TP+FN)/sum(C(:));
      P = (TP+FP)/sum(C(:));
      MCC(i) = (TP/sum(C(:))- S*P)/sqrt(S*P*(1-S)*(1-P));
      ERate(i) = 1-Acc(i);
     FPR(i) = 1-Spec(i);
end

disp('Accuracy .... ')

sprintf('The acc is : %2f', Acc)

disp('Sensitivity .... ')

sprintf('The sensitivity is : %2f', Sens)

disp('Specificity .... ')

sprintf('The specificity is : %2f', Spec)

disp('Precision .... ')

sprintf('The precision is : %2f', Prec)

disp('F1 score .... ')

sprintf('The f1score is : %2f', F1Sc)

%% Features extraction based on the last average pooling layer of ResNet for the training set 
trainingSet.ReadFcn = @(filename)readAndPreprocessImage(filename);
featureLayer =  'avg_pool';
trainingFeatures = activations(net1, trainingSet, featureLayer, ...
    'MiniBatchSize', 32, 'OutputAs', 'columns');

%% Features extraction based on the last average pooling layer of ResNet for the testing set 
% Extract testing set features using the CNN
testing.ReadFcn = @(filename)readAndPreprocessImage(filename);
testingFeatures = activations(net1,testing, featureLayer, ...
    'MiniBatchSize', 32, 'OutputAs', 'columns');
