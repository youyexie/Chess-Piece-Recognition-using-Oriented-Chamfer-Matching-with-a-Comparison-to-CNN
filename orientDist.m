function dist = orientDist(Gx_I,Gy_I,ADT,T,Gx_T,Gy_T,r0,c0)
       
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Input:  Gx_I, Gy_I - The directional gradients of the board image
%         ADT - The indexes of the nearest edge points
%         T - The edge image of the template
%         Gx_T, Gy_T - The directional gradients of the template
%         r0,c0 - Current row and column 
% Output: dist - The oriented chamfer score
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[rows,cols] = find(T);

linearIndices_T = sub2ind(size(ADT), rows+r0-1,cols+c0-1);

% Get locations of corresponding closest points in I.
linearIndices_I = ADT(linearIndices_T);

% Get orientations.
ux_T = Gx_T(T(:));
uy_T = Gy_T(T(:));
ux_I = Gx_I(linearIndices_I);
uy_I = Gy_I(linearIndices_I);

% Take dot product between vectors.
dp = ux_T.*ux_I + uy_T.*uy_I;

angDiff = acos(abs(dp));
if ~isreal(angDiff)
    angDiff = real(angDiff);
end

dist = sum(angDiff);

% %%%%%%%%%%%%%%%%%%%%%%%
% % This was just to check whether the above calculation was correct.
% distances = [];
% for xt=1:size(T,2)
%     for yt=1:size(T,1)
%         if T(yt,xt) == 0   continue;    end
%         
%         vT = [Gx_T(yt,xt); Gy_T(yt,xt)];  % orientation of T
%         
%         % Get corresponding point in I.
%         xi = xt-(w2+1) + c0;
%         yi = yt-(h2+1) + r0;
%         [py,px] = ind2sub(size(ADT), ADT(yi,xi));
%         
%         vI = [Gx_I(py,px); Gy_I(py,px)];
%         diff = acos(abs(dot(vT,vI)));
%         if ~isreal(diff)
%             diff = real(diff);
%         end
%         distances = [distances diff];
%     end
% end
% dist = sum(distances);
% fprintf('dist = %f\n', dist);

return

