function [t_enc,t_off] = my_pca(image)
[height,width,channel] = size(image);
% 把图像不同通道分别拉成一列
image = reshape(image,[height*width,channel]);
% 把图像划分为8*8的格子计算均值，并得到减去均值的新图片
for i=1:64:height*width
    mean_image = mean(image(i:i+63,:),1);
    de_mean_image(i:i+63,:) = image(i:i+63,:) - repmat(mean_image,[64,1]);
end
% 用新图片计算协方差矩阵
cov_matrix = (de_mean_image'*de_mean_image)/(height*width);
% 分解得到特征值和特征向量
[V,D] = eig(cov_matrix);
[~,index] = sort(diag(D),'descend');
d_matrix = V(:,index(1:3))';
% 按照论文来规整化
t_enc(1,:) = d_matrix(1,:)/norm(d_matrix(1,:),1)*219/255;
t_enc(2,:) = d_matrix(2,:)*(224/255/(sum(abs(d_matrix(2,:)))));
t_enc(3,:) = d_matrix(3,:)*(224/255/(sum(abs(d_matrix(3,:)))));
t_off(1,:) = 16;
mask = t_enc(2,:);
t_off(2,:) = -1*sum(mask(mask<0))*255 + 16;
mask = t_enc(3,:);
t_off(3,:) = -1*sum(mask(mask<0))*255 + 16;
end