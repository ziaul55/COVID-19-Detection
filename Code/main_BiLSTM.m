
%% This research is made available to the research community.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% If you are using this code please cite the following paper:                                              %
% Muhammad, U., Hoque, M.Z., Oussalah, M., Keskinarkaus, A., Seppänen, T. and Sarder, P., 2022.             % 
% SAM: Self-augmentation mechanism for COVID-19 detection using chest X-ray images. Knowledge-Based Systems % 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% BiLSTM

% RICA BASED BiLSTM
% Changing RICA features into BiLSTM format
rng(1);
trainf = {};
trainf{end+1} =  train1'; % input features from RICA augmentation 1
trainf{end+1} =  train2'; % input features from RICA augmentation 2
trainf{end+1} =  train3'; % input features from RICA augmentation 3

% converting labels
trainlabl = {};
trainlabl{end+1} = trainingLabels'; % training labels
trainlabl{end+1} = trainingLabels'; % training labels 
trainlabl{end+1} = trainingLabels'; % training labels

% feature size according to RICA output
numFeatures = 500;
% Setting the hidden layer
numHiddenUnits = 100;
% Setting the no. of classes
numClasses = 2;

% Specify the layer 
layers = [ ...
    sequenceInputLayer(numFeatures)
    bilstmLayer(numHiddenUnits, 'OutputMode','sequence', 'RecurrentweightsInitializer','he')
    fullyConnectedLayer(numClasses,'WeightsInitializer','he')
    softmaxLayer
    classificationLayer];
% Setting the learning rate
options = trainingOptions('adam', ...
    'ExecutionEnvironment','gpu', ... 
    'InitialLearnRate',0.0001, ...
    'MaxEpochs',150, ...
    'Verbose',0, ...
    'Plots','training-progress');
% Training the BiLSTM network
lstm2 = trainNetwork(trainf,trainlabl ,layers,options);

% classify the testing set
[predictedLabels, scores4] = classify(lstm2, test1');

% Get the known labels
testLabels = testing.Labels;

% Tabulate the results using a confusion matrix.
confMat = confusionmat(testLabels', predictedLabels);

% Convert confusion matrix into percentage form
confMat = bsxfun(@rdivide,confMat,sum(confMat,2));

%% Compute accuracy, Sensitivity, Specificity, Precision and F1 score by the following code
YValidation =  testLabels;
%accuracy4 = mean(predictedLabels == YValidation);

CPredicted=predictedLabels;

C = confusionmat(YValidation,CPredicted);

OverallAccTst4 = sum(YValidation==CPredicted)/length(CPredicted);
              
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
