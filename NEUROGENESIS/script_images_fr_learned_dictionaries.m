is_mairal = false;
% 
k=500;
% 
if is_mairal
    D = model.mairal.D{k};
    A = model.mairal.A{k};
else
    D = model.neurogen_group_mairal.D{k};
    A = model.neurogen_group_mairal.A{k};
end
% 
d = diag(A);
[~, idx] = sort(-d);
count=0;
suffix = '_nonsparsedict.png';
% 
for i=idx(1:25)' 
    count = count + 1;
    I = reshape(D(:,i), [100 100]);
    pcolor(I);
    shading interp;
    colorbar;
    saveas(gcf, strcat('pcolor_', strcat(num2str(count), suffix)));
    %     
    I = I*255; 
    imwrite(I, strcat('image_', strcat(num2str(count), suffix)));
end
