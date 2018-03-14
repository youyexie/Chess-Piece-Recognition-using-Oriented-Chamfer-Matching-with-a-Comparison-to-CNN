function Intersection=getIntersection(corners,nx,ny,mapimport)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Input:  corners - The square index
%         nx, ny - number of intersections in x and y axes
%         mapimport - whether the corners are imported
% Output: Intersections - The square corners location
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Make sure that points are in clockwise order
v12 = corners(2,:) - corners(1,:);
v13 = corners(3,:) - corners(1,:);
if v12(1)*v13(2) - v12(2)*v13(1) < 0
    temp = corners(2,:);
    corners(2,:) = corners(4,:);
    corners(4,:) = temp;
end

% Let the point starts from left top
sortcorners = corners;

if mapimport == 0
    sortcorners=sortcorners(:,1).^2+sortcorners(:,2).^2;
    index = find(sortcorners==min(sortcorners));
    if length(index)>1
        index = index(1);
    end
    sortcorners = circshift(corners, [-(index-1), 0]);
end

% Find the block intersections
sizeSquare = 1;
[xIntersectionsRef, yIntersectionsRef] = meshgrid(1:nx, 1:ny);
xIntersectionsRef = (xIntersectionsRef-1)*sizeSquare + 1;
yIntersectionsRef = (yIntersectionsRef-1)*sizeSquare + 1;

PtsRef = [1 1;
    nx 1;
    nx ny;
    1 ny];
T = fitgeotrans(PtsRef,sortcorners , 'projective');

% Transform all reference points to the image.
Intersection = [xIntersectionsRef(:) yIntersectionsRef(:) ones(size(yIntersectionsRef(:),1),1) ];
tranIntersection = T.T'*Intersection';
Intersection = zeros(2,size(yIntersectionsRef(:),1));
Intersection(1,:)=tranIntersection(1,:)./tranIntersection(3,:);
Intersection(2,:)=tranIntersection(2,:)./tranIntersection(3,:);

return
