function Squarecenters = findSquarecenters(Intersections)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Input:  Intersections - The square corners location
% Output: Squarecenters - The center of the squares
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Squarecenters = zeros(2,64);

for i = 1:length(Squarecenters)
    squarecorners = getSquarecorners(i,Intersections);
    Squarecenters(:,i) = mean(squarecorners)';
end

return