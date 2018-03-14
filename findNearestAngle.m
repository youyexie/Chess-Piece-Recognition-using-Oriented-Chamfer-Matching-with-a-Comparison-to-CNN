function Nearestangle = findNearestAngle(angle,angvec)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Input:  angle - The angle we want to round
%         angvec - The angle values that we want to round to
% Output: Nearestangle - The nearest angle of 'angle' in angvec 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

temp = abs(angle - angvec);
index = find(temp == min(temp));

if length(index)>1
    index=index(1);
end

Nearestangle = angvec( index );

end