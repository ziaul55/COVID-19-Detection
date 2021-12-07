%% RICA AUGMENTATION

%% This research is made available to the research community.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% If you are using this code please cite the following paper:                                  %
% Muhammad, U., Hoque, M.Z., Oussalah, M., Keskinarkaus, A., Seppänen, T. and Sarder, P., 2021. % 
% A Self-Attention Mechanism for COVID-19 Detection using Chest X-Ray Images.                   % 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

 q = 500;
 Mdl = rica(trainingFeatures',q,'IterationLimit',80);
 train1 = transform(Mdl,trainingFeatures');
 test1 = transform(Mdl,testingFeatures');
   
 q = 500;
 Mdl = rica(trainingFeatures',q,'IterationLimit',120);
 train2 = transform(Mdl,trainingFeatures');
 test2 = transform(Mdl,testingFeatures');

 q = 500;
 Mdl = rica(trainingFeatures',q,'IterationLimit',100);
 train3 = transform(Mdl,trainingFeatures');
 test3 = transform(Mdl,testingFeatures');
