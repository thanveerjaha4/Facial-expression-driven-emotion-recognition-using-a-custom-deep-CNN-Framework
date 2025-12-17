%% Step 1: Load the digit dataset
% Create an image datastore object to automatically load images
% Include all subfolders, and use folder names (0–9) as labels
imds = imageDatastore("C:\Users\Thanveer Jaha\OneDrive\Desktop\AI project",'IncludeSubfolders',true,'LabelSource','foldernames');
% Count number of images in each class (0–9)
countEachLabel(imds)

%% Step 2: Define CNN architecture (layers)
layers = [imageInputLayer([48 48])
convolution2dLayer(3,8,'Padding','same')
batchNormalizationLayer
reluLayer
maxPooling2dLayer(2,'Stride',2)
convolution2dLayer(3,16,'Padding','same')
batchNormalizationLayer
reluLayer
convolution2dLayer(3,24,'Padding','same')
fullyConnectedLayer(7)
softmaxLayer
classificationLayer];
%% Step 3: Set training options
options = trainingOptions('sgdm', ... % Use Stochastic Gradient Descent with Momentum
'InitialLearnRate',0.01, ... % Learning rate for weight updates
'MaxEpochs',100,...% Train for 4 full passes over dataset
'Shuffle','every-epoch', ...
'minibatchsize',64, ...
... % Shuffle data each epoch to avoid bias
'Verbose',false, ... % Suppress detailed command window output
'Plots','training-progress'); % Show training progress (accuracy/loss curves)
%% Step 4: Train the CNN
% Train the network using the defined layers and training options
net = trainNetwork(imds,layers,options);
%% Step 5: Test/Validate the CNN
% Predict labels for all images using trained network
YPred = classify(net,imds);
% True labels from dataset
YValidation = imds.Labels;
%% Step 6: Evaluate performance
% Calculate accuracy = (correct predictions / total samples)
accuracy = sum(YPred == YValidation)/numel(YValidation)
% Display confusion matrix (true vs predicted labels)
confusionchart(YValidation,YPred)