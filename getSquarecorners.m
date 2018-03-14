function sqaurecorners = getSquarecorners(k,Intersections)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Input:  k - The square index
%         Intersections - The square corners location
% Output: squarecorners - The four corners of the current square
%         [1  3
%          2  4]
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Intersections = Intersections';
sqaurecorners = zeros(4,2);

row = k-(ceil(k/8)-1)*8;
col = ceil(k/8);

sqaurecorners(1,:)= Intersections((col-1)*9 + row,:);
sqaurecorners(2,:)= Intersections((col-1)*9 + row + 1,:);
sqaurecorners(3,:)= Intersections((col-1)*9 + row + 9,:);
sqaurecorners(4,:)= Intersections((col-1)*9 + row + 9 + 1,:);

return
