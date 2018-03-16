clear variables; close all; clc

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Ang (pan angle) = '00' '10' '20' '30'
%  Resolution = 120 240 360 480 720 960
%  lambda = scalar between 0 and 1 (d_score = (1-lambda)*d_dist + lambda*d_ori)
%  ShowPicture = true or false
%  Occlusion = scalar between 0 and 1 (0% - 100% occlusion)
%  Import_control = true or false (import the true piece location)
%  Sampling = the sampling stride during matching

% Parameters initialization
Ang = '00';
Resolution = 720;
lambda = 0.5;
ShowPicture = true;
Occlusion = 0.3;        
Import_control = false;
Sampling = 6;

assert(strcmp(Ang,'00') || Occlusion==0,'Only add occlusion when Ang = 00')

% The name of board
Board = strcat(Ang,'.jpg');

% Pieces recognition
Result = recognition(Board,Resolution,lambda,ShowPicture,Occlusion,Import_control,Sampling);

% Calculate the recognition accuracy
Accuracy = compare(Result);
fprintf('The recognition accuracy for this image is %.1f%%.\n',Accuracy*100);
