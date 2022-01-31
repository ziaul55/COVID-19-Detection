%% This research is made available to the research community.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% If you are using this code please cite the following paper:                                              %
% Muhammad, U., Hoque, M.Z., Oussalah, M., Keskinarkaus, A., Seppänen, T. and Sarder, P., 2022.             % 
% SAM: Self-augmentation mechanism for COVID-19 detection using chest X-ray images. Knowledge-Based Systems % 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all
close all
clc
%%  Input for COVID dataset
rootFolder = fullfile('D:\CoVid19 results\Processed dataset\X-Ray Image DataSet\example');
categories1 = {'COVID','non-COVID'};
trainingSet1 = imageDatastore(fullfile(rootFolder, categories1), 'IncludeSubfolders',true, ...
 'LabelSource','foldernames');
%% Split data for training and testing
[trainingSet, testing] = splitEachLabel(trainingSet1,0.8);
% Extract training labels
trainingLabels = trainingSet.Labels;

%% Model input
% Input for deep learning model (ResNet-50)
net = resnet50;
net.Layers(1);
% Specify layers
inputSize = net.Layers(1).InputSize;

if isa(net,'SeriesNetwork') 
  lgraph = layerGraph(net.Layers); 
else
  lgraph = layerGraph(net);
end

%% 
% freezing initial weights
[learnableLayer,classLayer] = findLayersToReplace(lgraph);
%[learnableLayer,classLayer]
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
% Setting learning rate

miniBatchSize = 32;
options = trainingOptions('sgdm', ...
    'ExecutionEnvironment','gpu', ... 
    'MiniBatchSize',miniBatchSize, ...
    'MaxEpochs',5, ...
    'InitialLearnRate',3e-4, ...
    'Shuffle','every-epoch', ...
    'Verbose',true, ...
     'Plots','training-progress');
% resize images according to model size
trainingSet.ReadFcn = @(filename)readAndPreprocessImage(filename);
% Train the model 
net1 = trainNetwork(trainingSet,lgraph,options);
%% Model testing 
% resize images for testing set
testing.ReadFcn = @(filename)readAndPreprocessImage(filename);
% predict labels on testing set
[predictedLabels, scores] = classify(net1 ,testing);
% Get the known labels
testLabels = testing.Labels;
%% Compute accuracy, Sensitivity, Specificity, Precision and F1 score by the following code

% rename variable name of test labels as YValidation
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
