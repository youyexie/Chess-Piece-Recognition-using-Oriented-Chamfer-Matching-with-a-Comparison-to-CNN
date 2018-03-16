function Result = recognition(Board,Resolution,lambda,ShowPicture,Occlusion,import_control,Sampling)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Ang (pan angle) = '00' '10' '20' '30'
%  Resolution = 120 240 360 480 720 960
%  lambda = scalar between 0 and 1 (d_score = (1-lambda)*d_dist + lambda*d_ori)
%  ShowPicture = true or false
%  Occlusion = scalar between 0 and 1 (0% - 100% occlusion)
%  Import_control = true or false (import the true piece location)

% Read the current folder
currentFolder = pwd;
% Load the board
dirNameBoard = strcat(currentFolder,'\Boards');

% Load the template
dirNameTem = strcat(currentFolder,'\Templates');




%% Load the board
I = imread(sprintf('%s/%s', dirNameBoard, Board));
size_control = Resolution/3024;% 3024*4032    
% 120p (120x160) 1/25.2; 240p (240x320) 1/12.6; 360p (360x480) 1/8.4;
% 480p (480x640) 1/6.3 ; 720p (702x960) 1/4.2 ; 
% 960p (960x1280) 1/3.15; 1080p (1080x1440) 1/2.8;

I = imresize(I,size_control);

IC=I;
if ShowPicture
    figure(1),imshow(IC);
end
if size(I,3) > 1
    I = rgb2gray(I);
end


%% Find the board or load the existing board
File = strcat(currentFolder,'\Data\Corners',Board,'.mat');
if exist(File, 'file')== 2
    corners = importdata(File);
    mapimport=1;
    corners = corners*size_control; % Resize the imported corners
    if ShowPicture
        figure(2);imshow(I)
        hold on
    end
else
    % Hand pick the corners (Note: Only handpick when size_control=1)
    figure(2);imshow(I)
    corners = zeros(4,2);
    hold on
    for i=1:4
        corners(i,:) = ginput(1);
        plot(corners(i,1),corners(i,2),'r*');
    end
    save(File,'corners');
end

% Find the rotation matrix R_w_c
[R,K,d,n] = findRotations(corners,I);

% Get the pieces intersections
nx=9; % Intersection points in x direction
ny=9; % Intersection points in y direction
Intersections = getIntersection(corners,nx,ny,mapimport);
Squarecenters = findSquarecenters(Intersections);

% Draw the boundary of the chessboard
if ShowPicture
    figure(2)
    line([corners(1,1) corners(2,1)],[corners(1,2) corners(2,2)],  'Color', 'g','Linewidth',2);
    line([corners(2,1) corners(3,1)],[corners(2,2) corners(3,2)],  'Color', 'g','Linewidth',2);
    line([corners(3,1) corners(4,1)],[corners(3,2) corners(4,2)],  'Color', 'g','Linewidth',2);
    line([corners(1,1) corners(4,1)],[corners(1,2) corners(4,2)],  'Color', 'g','Linewidth',2);
end
% The angles between the viewing angle and the chessboard plane
Angles = zeros(64,1);
Rotation = zeros(64,2); %(angle direction)



%% Draw coordinate axis vectors on the input image.
for i=1:length(Squarecenters)
    
    % Extract the squares' centers
    p = inv(K)*[Squarecenters(:,i) ; 1];       % Normalized image coordinates
    s = n' * p;                 % If p is on the plane, then s=d
    P = (d/s)*p;            % Now, P is on the plane
    
    % Project P onto the image.
    p0 = K*P;                   % origin
    p0 = p0/p0(3);
    
    % Make sure the z axis is pointing up
    if i==1
        uz = K*(P + R*[0;0;1]);     % z axis
        uz = uz/uz(3);
        if (p0(2)-uz(2))<0
            VecZ=[0;0;-1];
        else
            VecZ = [0;0;1];
        end
    end
    
    ux = K*(P + R*[1;0;0]);     % x axis
    ux = ux/ux(3);
    uy = K*(P + R*[0;1;0]);     % y axis
    uy = uy/uy(3);
    uz = K*(P + R*VecZ);     % z axis
    uz = uz/uz(3);
    
    if ShowPicture
        line([p0(1) ux(1)], [p0(2) ux(2)], 'Color', 'r', 'LineWidth', 2);
        line([p0(1) uy(1)], [p0(2) uy(2)], 'Color', 'g', 'LineWidth', 2);
        line([p0(1) uz(1)], [p0(2) uz(2)], 'Color', 'b', 'LineWidth', 2);
    end
    
    % Calculate the viewing angles
    a = R*VecZ;
    b = -P;
    Angles(i) =  atan2(norm(cross(a,b)),dot(a,b))*180/pi;
    
    % Calculate the rotation angles
    temp = uz-p0;
    a1=temp(1:2,:);
    a1 = a1/norm(a1); % Normalized
    a1(2)=-a1(2);
    b1 = [0;1];
    angle= acos(a1'*b1)*180/pi;
    
    if temp(1)>0
        direction = -1;
    else if temp(1)<0
            direction = 1;
        end
    end
    Rotation(i,1) = angle;
    Rotation(i,2) = direction;
end

% Draw the intersections
minD = min(size(I));
DMIN = minD/50;

if ShowPicture
    figure(2)
    for i=1:size(Intersections(1,:)',1)
        rectangle('Position', [Intersections(1,i)-DMIN/2 Intersections(2,i)-DMIN/2 DMIN DMIN], 'EdgeColor', 'g','Curvature',[1 1]);
    end
    hold off
end

%% Transform the detected board to ortho-view
Imask = zeros(320,320);
PtsMask = [1 1;
    320 1;
    320 320;
    1 320];
Pts1 = corners;
Tmask1 = fitgeotrans(Pts1,PtsMask , 'projective');
Iortho = imwarp(I, Tmask1, 'OutputView', ...
    imref2d(size(Imask), [1 size(Imask,2)], [1 size(Imask,1)]));

% Find the squares with pieces
[~,thresh] = edge(Iortho, 'canny');      % First get the automatic threshold
Eortho = edge(Iortho, 'canny',1.2*thresh);     % Raise the threshold
if ShowPicture
    figure(3);imshow(Eortho)
end

%% Determine the pieces locations
PiecesLocation = zeros(8,8);
if ShowPicture
    figure(3)
end
ds=15; % Detection size
shift=5; % Move upward
for i = 1:8
    for j = 1:8
        PiecesLocation(i,j) = sum(sum(Eortho(40*(i-1)+floor((40-ds)/2)-shift:40*(i-1)+floor((40-ds)/2)+ds-shift,40*(j-1)+5:40*(j-1)+35)));
        if ShowPicture
            rectangle('Position', [40*(i-1)+5 floor((40*(j-1))+floor((40-ds)/2))-shift 30 ds], 'EdgeColor', 'g');
        end
    end
end
PiecesLocation = ( PiecesLocation > 5 ); %0.5*mean(mean(PiecesLocation))

% Assumption: No piece is right behind another since the effect of
% occlusion will be studied individually
for i=1:7
    for j=1:8
        if PiecesLocation(9-i,j)==1
            PiecesLocation(9-i-1,j)=0;
        end
    end
end

% Load the true piece location if import_control is true
File = strcat(currentFolder,'\Data\FinalPiecesLocation.mat');
if exist(File, 'file')== 2 && import_control
    PiecesLocation = importdata(File);
    fprintf('Import the true pieces locations!\n')
end

%% Get the edge input image
I = adapthisteq(I);

% Cover part of the pieces
File = strcat(currentFolder,'\Data\Locations.mat');
if  exist(File, 'file')== 2
    Locations = importdata(File);
    % Count the number of detection block
    mvalue = [];
    for i=1:size(Locations,1)
        mvalue=[ mvalue;min(min(Locations{i,3}))];
    end
    
    File = strcat(currentFolder,'\Data\FinalPiecesLocation.mat');
    tempLocation = importdata(File);
    templist = find(tempLocation);
    
    for indexm = 1:length(mvalue)
        A = Locations{templist(indexm),1}; % Get the pieces starting point
        B = Locations{templist(indexm),2}; % Get the pieces template h&w
        C = Locations{templist(indexm),3}; % Get the pieces chamfer matching score
        [minVal, index] = min(C(:));
        [r1,c1] = ind2sub(size(C),index);
        a1 = A(1); a2 = A(2);
        w1 = B(2); h1 = B(1);
        %x1 = a1+c1+w1; y1 = a2+r1+h1; x2 = a1+c1;  %[x2 x1]
        xl = a1+c1;     xr = a1+c1+w1;
        yb = a2+r1+h1;  yt = a2+r1+round(h1*(1-Occlusion));
        % Modify the place based on resolution
        xl = round(xl*Resolution/720); xr = round(xr*Resolution/720);
        yb = round(yb*Resolution/720); yt = round(yt*Resolution/720);
        
        if Occlusion~=0
        I(yt:yb,xl:xr)=0;
        IC(yt:yb,xl:xr,:)=0;
        end
    end
end
figure(1),imshow(IC)


[E,threshCanny] = edge(I, 'canny');

[H,W] = size(I);

% Get the gradient
[Gx,Gy] = imgradientxy(I);
Gmag = (Gx.^2 + Gy.^2).^0.5;
Gmag(Gmag==0)=1;
Gx_I = Gx ./ Gmag;      % unit vector
Gy_I = Gy ./ Gmag;

[D,ADT] = bwdist(E);   % Distance transform.
tau = 30;       % Threshold distance
D = min(D,tau);

if ShowPicture
    figure(4); imshow(E,[])
end

%% Load the template
Locations = cell(size(PiecesLocation(:),1),5);
% % Store the search result
% [Matching start point] [h,w of the template] [chamfer score map] [model name]

count = 0;

Sample=1;
fprintf('Sampling is:');
if Sample
    fprintf('On\n');
else
    fprintf('Off\n');
end

fprintf('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Matching start %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n');
Models={'King','Queen','Horse','Rook','Bishop','Pawn'}; % The names of models we used
tic;
for k=size(PiecesLocation(:),1):-1:1
    
    % Only search the squares with pieces
    [r,c]=ind2sub(size(PiecesLocation),k);%k-(ceil(k/8)-1)*8 , ceil(k/8)
    if PiecesLocation(r,c)==0
        continue
    end
    
    % Record the index of AOI
    count = count + 1;
    
    % Measure the square's parameters
    sqaurecorners = getSquarecorners(k,Intersections);  %[ 1  3 ]
                                                        %[ 2  4 ]
    xmax = max(sqaurecorners(:,1));
    xmin = min(sqaurecorners(:,1));
    ymax = max(sqaurecorners(:,2));
    ymin = min(sqaurecorners(:,2));
    
    Cminbest=10;
    
    % Change the aspect ratio respectively
    factor = 0;
    if r>=7 factor = 2;
    elseif r>=5 factor = 2.5;
    elseif r>=3 factor = 3;
    else factor = 3.5;
    end
    
    for i=1:length(Models)
        
        xrange = round(xmax-xmin);
        yrange = round(factor*(ymax-ymin));
        % Enlarge the matching area if the model is king or queen
        if strcmp(Models{i},'King') || strcmp(Models{i},'Queen')
            yrange = round(yrange*(factor+1)/factor);
        end
        
        % Select the model with right angle
        Nearestangle = findNearestAngle(Angles(k),[10,15,20,25,30,35,40,45,50,55,60,70]);
        
        % Load the corresponding template image
        It = imread(sprintf('%s/%s', dirNameTem, [Models{i},int2str(Nearestangle),'.jpg']));
        if size(It,3) > 1
            It = rgb2gray(It);
        end
        
        % Resize the template according to the size of the square
        rotate = 1.5;
        T = imresize(It,norm(((sqaurecorners(1,:)+sqaurecorners(2,:))/2 - (sqaurecorners(3,:)+sqaurecorners(4,:))/2 ))/size(It,2));
        Ttemp = imrotate(T,Rotation(k,2)*Rotation(k,1)/rotate );
        
        [ET,threshCanny] = edge(T, 'canny');
        
        [h,w] = size(T); % h w will be replaced
        edgeFraction=0.04;  %720 960 0.04 % 480 0.05 % 360 0.07 % 240 0.095 %120 0.12
        switch H
            case 120
                edgeFraction = 0.12;
                %fprintf('edgeFraction = 0.12\n');
            case 240
                edgeFraction = 0.095;
                %fprintf('edgeFraction = 0.095\n');
            case 360
                edgeFraction = 0.07;
                %fprintf('edgeFraction = 0.07\n');
            case 480
                edgeFraction = 0.05;
                %fprintf('edgeFraction = 0.05\n');
            case 720
                edgeFraction = 0.04;
                %fprintf('edgeFraction = 0.04\n');
            case 960
                %fprintf('edgeFraction = 0.04\n');
                edgeFraction = 0.04;
            case 1080
                %fprintf('edgeFraction = 0.04\n');
                edgeFraction = 0.03;
            otherwise
                edgeFraction = 0.04;
                %fprintf('Check your resolution!0.04(Default) is used.');
        end
        while sum(ET(:)) > 1.1*edgeFraction*h*w
            threshCanny = 1.1*threshCanny;
            if(threshCanny(2))>=1 threshCanny(2)=0.999;end
            ET = edge(T, 'canny', threshCanny);
        end
        while sum(ET(:)) < 0.9*edgeFraction*h*w
            threshCanny = 0.9*threshCanny;
            if(threshCanny(2))>=1 threshCanny(2)=0.999;end
            ET = edge(T, 'canny', threshCanny);
        end
        
        % Rotate the template based on the normal vector on that square (Rotation angle/2)
        ET = imrotate(ET,Rotation(k,2)*Rotation(k,1)/rotate );
        
        % Delete the redundant boundary
        [htemp, wtemp] = size(ET);
        [row,col]=ind2sub(size(ET),find(ET==1));
        minCol = col(1);
        maxCol = col(end);
        
        minRow = min(row);
        maxRow = max(row);
        
        %cut the redundant black area. size/20
        ET = ET(max(minRow-round(htemp/20),1):min(maxRow+round(htemp/20),htemp),max(minCol-round(wtemp/20),1):min(maxCol+round(wtemp/20),wtemp));
        
        Ttemp = Ttemp(max(minRow-round(htemp/20),1):min(maxRow+round(htemp/20),htemp),max(minCol-round(wtemp/20),1):min(maxCol+round(wtemp/20),wtemp));
        
        % Compute gradient of template
        [Gx,Gy] = imgradientxy(Ttemp);
        Gmag = (Gx.^2 + Gy.^2).^0.5;
        Gmag(Gmag==0)=1;
        Gx_T = Gx ./ Gmag;      % unit vector
        Gy_T = Gy ./ Gmag;
        
        T = ET;
        
        [h,w] = size(T);
        
        if ShowPicture
            figure(5);imshow(T,[]); % Show the templates
        end
        StartPoint = round([xmin ymax-yrange]);
        % Just in case the matching area goes outside the image
        if StartPoint(2)<0
            StartPoint(2)=1;
        end
        C = ones(yrange-h+1,xrange-w+1);      % Chamfer matching score image
        
        Jump=0;
        
        % Do the Chamfer matching
        for r=StartPoint(2):(StartPoint(2)+yrange-h)
            for c=StartPoint(1):(StartPoint(1)+xrange-w)
                
                
                % Sampling
                if Sample
                    Jump = Jump + 1;
                    if Jump <= (Sampling-1)
                        continue
                    end
                end
                %fprintf('Matching the %dth row and %dth column\n',r,c);
                % This is the distance score.
                d_cham = chamDist(D,T,r,c) / (tau * sum(T(:)));
                % This is the orientation score.
                d_orient = orientDist(Gx_I,Gy_I,ADT,T,Gx_T,Gy_T,r,c) / ((pi/2) * sum(T(:)));
                % The final score is a weighted sum of the two.
                C(r-StartPoint(2)+1,c-StartPoint(1)+1) = (1-lambda)*d_cham + lambda*d_orient;
                
                if Sample
                    if C(r-StartPoint(2)+1,c-StartPoint(1)+1)>=0.3
                        Jump=0;
                    end
                end
                
            end
        end
        
        if min(min(C)) < Cminbest
            
            Cminbest = min(min(C));
            
            Locations{k,1} = StartPoint; %(x,y)
            Locations{k,2} = [h,w];
            Locations{k,3} = C;
            Locations{k,4} = Models{i};
            Locations{k,5} = [xrange yrange];
        else
            continue
        end
        
    end
    
    
    fprintf('The %d th AOI lowest chamfer score:%.3f and winner is %s\n',count,Cminbest,[Locations{k,4},int2str(Nearestangle)]);
    
    % Filtered out the result whose score is too high in the next section
    DetectionThreshold=0.20;
    if Cminbest>=DetectionThreshold
        fprintf('However, the score is larger than the threshold.\n');
    end
    
    if ShowPicture
        figure(4)
        rectangle('Position', round([Locations{k,1} Locations{k,5}]), 'EdgeColor', 'g','Linewidth',2);  % The matching area
    end
    
end
fprintf('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Matching end %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n');

fprintf('Matching takes %d min %d seconds.\n',floor(toc/60),round(mod(toc,60)));

%% Filtering out the overlapped and large score detection, determine the final locations

if exist(File, 'file')== 2 && import_control
    FinalPiecesLocation = PiecesLocation;
else
    
    fprintf('Filtering out the overlapped and large score detection!\n');
    % Eliminate the overlapped detection result; 1 - overlapped, 2 - up
    
    
    if sum(PiecesLocation(:))==0
        fprintf('No pieces on the board!\n');
    else
        
        % record the minimum chamfer matching score
        mvalue = [];
        for i=1:size(Locations,1)
            mvalue=[ mvalue;min(min(Locations{i,3}))];
        end
        
        % templist records the piceces locations indexes.
        templist = find(PiecesLocation);
        FinalPiecesLocation = PiecesLocation;
        for indexm = 1:length(mvalue)
            
            A = Locations{templist(indexm),1}; % Get the pieces starting point
            B = Locations{templist(indexm),2}; % Get the pieces template h&w
            C = Locations{templist(indexm),3}; % Get the pieces chamfer matching score
            [minVal, index] = min(C(:));
            del=0;
            
            if minVal>=DetectionThreshold %  Filter out the result with high chamfer score
                Locations{templist(indexm),1}=[];
                Locations{templist(indexm),2}=[];
                Locations{templist(indexm),3}=[];
                Locations{templist(indexm),4}=[];
                Locations{templist(indexm),5}=[];
                FinalPiecesLocation(templist(indexm)) =0;
            else
                
                
                % Filter out the overlapped detection mainly for king and
                % queen who may occupy 3 squares.
                if ismember(templist(indexm)+2,templist)
                    A = Locations{templist(indexm),1}; % Get the pieces starting point
                    B = Locations{templist(indexm),2}; % Get the pieces template h&w
                    C = Locations{templist(indexm),3}; % Get the pieces chamfer matching score
                    [minVal, index] = min(C(:));
                    [r1,c1] = ind2sub(size(C),index);
                    a1 = A(1); a2 = A(2);
                    w1 = B(2); h1 = B(1);
                    x1 = a1+c1+w1; y1 = a2+r1+h1; x2 = a1+c1;  %[x2 x1]
                    
                    
                    A = Locations{templist(indexm)+2,1}; % Get the pieces starting point
                    B = Locations{templist(indexm)+2,2}; % Get the pieces template h&w
                    C = Locations{templist(indexm)+2,3}; % Get the pieces chamfer matching score
                    [minVal, index] = min(C(:));
                    [r,c] = ind2sub(size(C),index);
                    xl = A(1)+c; xr = A(1)+c+B(2);  % y-top y-bottom x-left x-right
                    yt = A(2)+r; yb = A(2)+r+B(1);
                    
                    if (y1>yt) & (y1<yb) & (x2>xl) & (x2<xr) & ( ( (xr-x2)*(y1-yt)/(w1*h1) )> 0.7)  % Left corners in it
                        Locations{templist(indexm),1}=[];
                        Locations{templist(indexm),2}=[];
                        Locations{templist(indexm),3}=[];
                        Locations{templist(indexm),4}=[];
                        Locations{templist(indexm),5}=[];
                        FinalPiecesLocation(templist(indexm)) =0;
                        del=1;
                    end
                    
                    if (y1>yt) & (y1<yb) & (x1>xl) & (x1<xr) & ( ( (x1-xl)*(y1-yt)/(w1*h1) )> 0.7 )  % Right corners in it
                        Locations{templist(indexm),1}=[];
                        Locations{templist(indexm),2}=[];
                        Locations{templist(indexm),3}=[];
                        Locations{templist(indexm),4}=[];
                        Locations{templist(indexm),5}=[];
                        FinalPiecesLocation(templist(indexm)) =0;
                        del=1;
                    end
                end
                
                if del
                    continue
                end
                
                % Filter out the flase position caused by 
                A = Locations{templist(indexm),1}; % Get the pieces starting point
                B = Locations{templist(indexm),2}; % Get the pieces template h&w
                C = Locations{templist(indexm),3}; % Get the pieces chamfer matching score
                [minVal, index] = min(C(:));
                [r1,c1] = ind2sub(size(C),index);
                a1 = A(1); a2 = A(2);
                w1 = B(2); h1 = B(1);
                x1 = a1+c1+w1; y1 = a2+r1+h1; x2 = a1+c1;  %[x2 x1]
                
                % Last square
                sqaurecorners = getSquarecorners(templist(indexm)-1,Intersections);  %[ 1  3 ]
                                                                                     %[ 2  4 ]
                xmax = max(sqaurecorners(:,1));
                xmin = min(sqaurecorners(:,1));
                ymax = max(sqaurecorners(:,2));
                ymin = min(sqaurecorners(:,2));
                yint = ymax-ymin;
                
                
                if  y1<(ymax+yint/5) & (ymax+yint/5-y1)<yint
                    Locations{templist(indexm)-1,1}= Locations{templist(indexm),1};
                    Locations{templist(indexm)-1,2}= Locations{templist(indexm),2};
                    Locations{templist(indexm)-1,3}= Locations{templist(indexm),3};
                    Locations{templist(indexm)-1,4}= Locations{templist(indexm),4};
                    Locations{templist(indexm)-1,5}= Locations{templist(indexm),5};
                    Locations{templist(indexm),1}=[];
                    Locations{templist(indexm),2}=[];
                    Locations{templist(indexm),3}=[];
                    Locations{templist(indexm),4}=[];
                    Locations{templist(indexm),5}=[];
                    FinalPiecesLocation(templist(indexm)) =0;
                    FinalPiecesLocation(templist(indexm)-1) =1;
                end
            end
        end
    end
end


%% Detect the color of each piece
fprintf('Detecting the color of each piece!\n');
%  White - 1
%  Black - 2
Color = 2.*(checkerboard(1)>0.5)+(checkerboard(1)==0); % Create the color matrix

% Extrarct one black and one white intensity templates
Intensity =zeros(1,8);
for r=8:-1:1
    for c=8:-1:1
        Intensity(r,c) = mean(mean( Iortho( 40*(r-1)+10:40*(r-1)+30 , 40*(c-1)+10: 40*(c-1)+30 ) ));
    end
end

BlackIntensity = 0;
WhiteIntensity = 0;
for i=1:8
    if (Color(8,i)==2) & (FinalPiecesLocation(8,i)==0)
        BlackIntensity = BlackIntensity + Intensity(8,i);
    elseif (Color(8,i)==1) & (FinalPiecesLocation(8,i)==0)
        WhiteIntensity = WhiteIntensity + Intensity(8,i);
    end
end
BlackIntensity = BlackIntensity/4;
WhiteIntensity =  WhiteIntensity/4;


% The locations of pieces
templist = find(FinalPiecesLocation);
for i=1:sum(sum(FinalPiecesLocation))
    % Find the origininal color, r+c=odd black; r+c=even white
    [r,c] = ind2sub(size(FinalPiecesLocation),templist(i));
    
    if mod(r+c,2) & Intensity(templist(i)) > BlackIntensity % White piece on black square
        Color(templist(i)) = Color(templist(i))-1;
    elseif mod(r+c,2)==0 & Intensity(templist(i)) < 0.7*WhiteIntensity % Black piece on white square
        Color(templist(i)) = Color(templist(i))+1;
    end
end
PiecesColor = Color .* FinalPiecesLocation;
%  White - 1
%  Black - 2


%% Show the final result
if ShowPicture
    
    fprintf('Showing the results.\n')
    % Count the number of detection block
    mvalue = [];
    for i=1:size(Locations,1)
        mvalue=[ mvalue;min(min(Locations{i,3}))];
    end
    
    templist = find(FinalPiecesLocation);
    % Insert text
    for indexm = 1:length(mvalue)
        
        % Show the name
        sqaurecorners = getSquarecorners(templist(indexm),Intersections);  %[ 1  3 ]
        %[ 2  4 ]
        xmax = max(sqaurecorners(:,1));
        xmin = min(sqaurecorners(:,1));
        ymax = max(sqaurecorners(:,2));
        ymin = min(sqaurecorners(:,2));
        
        if PiecesColor(templist(indexm))==2
            color = 'B-';
        else color = 'W-';
        end
        IC=insertText(IC,round([xmin+(xmax-xmin)/6,ymax]),strcat(color,Locations{templist(indexm),4}),'FontSize',round(60*size_control));
        
    end
    
    
    % Show the board detection result first.
    figure(6);imshow(IC,[]);%title('Recognition result')
    line([corners(1,1) corners(2,1)],[corners(1,2) corners(2,2)],  'Color', 'g','Linewidth',2);
    line([corners(2,1) corners(3,1)],[corners(2,2) corners(3,2)],  'Color', 'g','Linewidth',2);
    line([corners(3,1) corners(4,1)],[corners(3,2) corners(4,2)],  'Color', 'g','Linewidth',2);
    line([corners(1,1) corners(4,1)],[corners(1,2) corners(4,2)],  'Color', 'g','Linewidth',2);
    for i=1:size(Intersections(1,:)',1)
        rectangle('Position', [Intersections(1,i)-DMIN/2 Intersections(2,i)-DMIN/2 DMIN DMIN], 'EdgeColor', 'g','Curvature',[1 1]);
    end
    
    
    if sum(FinalPiecesLocation(:))==0
        fprintf('No pieces on the board!\n');
    else
        for indexm = 1:length(mvalue)
            
            
            A = Locations{templist(indexm),1}; % Get the pieces starting point
            B = Locations{templist(indexm),2}; % Get the pieces template h&w
            C = Locations{templist(indexm),3}; % Get the pieces chamfer matching score
            
            [minVal, index] = min(C(:));
            
            [r,c] = ind2sub(size(C),index);
            %     fprintf('Minimum value = %f at (r,c)=(%d,%d)\n', minVal, r+A(2)-1,c+A(1)-1);
            
            
            figure(4)
            rectangle('Position', [c+A(1)-1 r+A(2)-1 B(2) B(1)], 'EdgeColor', 'r','Linewidth',2);
            
            figure(6)
            rectangle('Position', [c+A(1)-1 r+A(2)-1 B(2) B(1)], 'EdgeColor', 'r','Linewidth',2);
            
            
        end
    end
end


%% Return the final detection result in a matrix form
Result = zeros(8,8);
for i=1:64
    if FinalPiecesLocation(i)==1
        a = Locations{i,4};
        switch a
            case 'King'
                Result(i)=1;
            case 'Queen'
                Result(i)=2;
            case 'Bishop'
                Result(i)=3;
            case 'Horse'
                Result(i)=4;
            case 'Rook'
                Result(i)=5;
            case 'Pawn'
                Result(i)=6;
        end
    end
end







