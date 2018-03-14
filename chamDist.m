function dist = chamDist(D,T,r0,c0)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Input:  D - Distance transformed image of the board
%         T - The template edge image
%         r0,c0 - Current row and column 
% Output: dist - The chamfer distance score
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


[rows,cols] = find(T);

linearIndices = sub2ind(size(D), rows+r0-1,cols+c0-1);
distances = D(linearIndices);
dist = sum(distances);

return

