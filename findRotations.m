% Estimate rotation matrix from vanishing points.
% See Szeliski book, section 6.3.2.

function [R,K,d,n] = findRotations(corners,I)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Input:  corners - the four detected chessboard corners
%         I - input image
% Output: R - This is the rotation of world-to-camera, R_w_c
%         K - Camera intrinsic matrix
%         d - Hyperplane factor
%         n - Hyperplane normal vector
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Find the vanishing point in the x axis direction.
% If the user clicked on the points in the correct order, this should be
% defined by the two line segments:  pt1 to pt2 and pt4 to pt3.

pts = corners;

% The equation of the line between pt1 and pt2 is pt1 cross pt2.
x1 = pts(1,1); y1 = pts(1,2);
x2 = pts(2,1); y2 = pts(2,2);
l1 = cross([x1;y1;1], [x2;y2;1]);
%fprintf('1st x line is (a,b,c) = (%f,%f,%f), where ax + by + c = 0\n', ...
%    l1(1), l1(2), l1(3));

% The equation of the line between pt4 and pt3 is pt4 cross pt3.
x4 = pts(4,1); y4 = pts(4,2);
x3 = pts(3,1); y3 = pts(3,2);
l2 = cross([x4;y4;1], [x3;y3;1]);
%fprintf('2nd x line is (a,b,c) = (%f,%f,%f), where ax + by + c = 0\n', ...
%    l2(1), l2(2), l2(3));

% Two lines l1 and l2 intersect    at a point p where p = l1 cross l2.
px = cross(l1,l2);      % Could be off the image
px = px/px(3);
fprintf('x vanishing point is at (x,y) = (%f,%f)\n', px(1), px(2));
% rectangle('Position', [px(1)-5 px(2)-5 10 10], 'EdgeColor', 'r');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Find the vanishing point in the y axis direction.
% If the user clicked on the points in the correct order, this should be
% defined by the two line segments:  pt2 to pt3 and pt1 to pt4.

% The equation of the line between pt2 and pt3 is pt2 cross pt3.
x2 = pts(2,1); y2 = pts(2,2);
x3 = pts(3,1); y3 = pts(3,2);
l1 = cross([x2;y2;1], [x3;y3;1]);
%fprintf('1st y line is (a,b,c) = (%f,%f,%f), where ax + by + c = 0\n', ...
%    l1(1), l1(2), l1(3));

% The equation of the line between pt1 and pt4 is pt1 cross pt4.
x1 = pts(1,1); y1 = pts(1,2);
x4 = pts(4,1); y4 = pts(4,2);
l2 = cross([x1;y1;1], [x4;y4;1]);
%fprintf('2nd y line is (a,b,c) = (%f,%f,%f), where ax + by + c = 0\n', ...
%    l2(1), l2(2), l2(3));

% Two lines l1 and l2 intersect at a point p where p = l1 cross l2.
py = cross(l1,l2);      % Could be off the image
py = py/py(3);
fprintf('y vanishing point is at (x,y) = (%f,%f)\n', py(1), py(2));
% rectangle('Position', [py(1)-5 py(2)-5 10 10], 'EdgeColor', 'r');



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Estimate focal length
% We use the method in Szeliski's book on Computer Vision (section 6.3.2,
% page 330).  This says that
% the two vanishing points represent two orthogonal directions.  The
% directions are [px(1)-cx; px(2)-cy; f] and [py(1)-cx; py(2)-cy; f], where
%  cx,cy is the center of the image (assumed in the middle) and f is focal
%  length.  So the dot product is zero:
% (px(1)-cx)(py(1)-cx) + (px(2)-cy)(py(2)-cy) + f^2 = 0
cx = size(I,2)/2;
cy = size(I,1)/2;
dp = (px(1)-cx)*(py(1)-cx) + (px(2)-cy)*(py(2)-cy);
if dp >= 0
    fprintf('Warning ... focal length undefined; just picking one\n');
    f = min(cx,cy);
else
    f = sqrt( -dp );
end
fprintf('Estimated focal length = %f pixels\n', f);
K = [f  0  cx;  0  f  cy;  0  0  1];    % camera intrinsic params



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Get full rotation matrix
R = zeros(3,3);     % This is the rotation of world-to-camera, R_w_c

vx = [px(1)-cx; px(2)-cy; f];
vy = [py(1)-cx; py(2)-cy; f];
vx = vx/norm(vx);   % Make unit vectors
vy = vy/norm(vy);

vz = cross(vx, vy);    % Get +Z axis
R(:,1) = vx;
R(:,2) = vy;
R(:,3) = vz;
% Actually, this may not result in a valid rotation matrix due to noise.
% Should really implement Szeliski's recommendation on finding R using the
% absolute orientation algorithm (also see page 330 in his book).
%fprintf('Estimated rotation matrix, world to camera:\n');
%disp(R);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The equation of a plane is n*P = d, where n is the normal and d is the
% perpendicular distance.
n = vz;
% Let's just say that point 0,0,10 (in camera coords) is on the plane.
d = n' * [0;0;10];

end