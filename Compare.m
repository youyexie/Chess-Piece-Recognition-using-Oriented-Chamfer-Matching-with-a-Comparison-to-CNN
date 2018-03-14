function Accuracy = Compare(Result)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Input:  Result - The recognition result in the matrix form
% Output: Accuracy - The recognition accuracy
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Load the ground truth
currentFolder = pwd;
File = strcat(currentFolder,'\Data\Groundtrue.mat');
GroundTrue = importdata(File);
temp = 0 ;

% Loop over to compare
for i=1:64
    if Result(i)~=0 && Result(i)== GroundTrue(i)
        temp = temp + 1;
    end
end

% Calculate the accuracy
Accuracy = temp/sum(sum(GroundTrue>0));

return
