%this is the main script of how to run the SPIHT and DWT algorithm
% clc
% clear all

% FileName='Lena.tif';
% Orig_I1=imread(FileName);
% image_small=double(imread('lena_small.tif'));

image_small = double(imread('/Users/zoe/desktop/images/lena_small.tif'));
Orig_I1       = imread('/Users/zoe/desktop/images/lena.tif');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% original
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% version
% %color space transforms
%  yy=rgb2ycbcr(Orig_I1);
% %[t_enc,t_off] = my_pca(Orig_I1); % pca version:
% 
% Orig_I2 = yy;
% Orig_I2_small = rgb2ycbcr(image_small);
% %Note some useful Imformation
% fl_dp=8;%it changes after PCA
% [~,~,Layers]= size(Orig_I1);%checks the type of the oringinal Image, grey or color
% index=0;
% for rate=0.1:0.1:1.5%different compressions ration, like qScale
%     index=index+1;
%     len=0;
%     for ij=1:3%do the compression each layer
% 
%         fprintf('plane %d computing\n\n',ij);
% 
%         Orig_I=Orig_I2(:,:,ij);
%         Orig_I_small=Orig_I2_small(:,:,ij);
%         %preparation for each maxbit of SPIHT Algorithm
%         if ij==2
%             [s_1,s_2] = size(Orig_I);
%             s_1=ceil(s_1/2);
%             s_2=ceil(s_2/2);
% 
%             [ss_1,ss_2] = size(Orig_I_small);
%             ss_1=ceil(ss_1/2);
%             ss_2=ceil(ss_2/2);
%         elseif ij==3
%             [s_1,s_2] = size(Orig_I);
%             s_1=ceil(s_1/2);
%             s_2=ceil(s_2/2);
% 
%             [ss_1,ss_2] = size(Orig_I_small);
%             ss_1=ceil(ss_1/2);
%             ss_2=ceil(ss_2/2);
% 
%         else
%             [s_1,s_2] = size(Orig_I);
%             [ss_1,ss_2] = size(Orig_I_small);
%         end
%         max_bits_s = floor(rate * (ss_1*ss_2));
%         max_bits = floor(rate * (s_1*s_2));
%         %OutSize = s_1;
%         %image_spiht = zeros(ceil(size(Orig_I)/2));
%         [nRow, nColumn] = size(Orig_I);
%         [nRow_s, nColumn_s] = size(Orig_I_small);
%          [n_1,n_2] = size(Orig_I);
%          [n_1_s,n_2_s]=size(Orig_I_small);
%         if n_1<n_2
%             n_log = log2(n_2);
%             n_log_s = log2(n_2_s);
%         else
%             n_log = log2(n_1);
%             n_log_s = log2(n_1_s);
%         end
%         level =floor(n_log);
%         level_s =floor(n_log_s);
%         %% do dwt encoding
%         type='bior4.4';%JEPG 2000
%         [Lo_D,Hi_D,Lo_R,Hi_R] = wfilters(type);%??4 filters for dwt,this case haar transform
% 
%         [I_W, S] = func_DWT(Orig_I, level, Lo_D, Hi_D);
%         [I_W_S,S1]=func_DWT(Orig_I_small,level_s, Lo_D, Hi_D);
%         I_W_3(:,:,ij)=uint8(I_W);
% %         figure
% %         imshow(I_W_3)
%         % Use SPIHT to compress the bit stream
%         img_enc = func_SPIHT_Enc(I_W, max_bits, (nRow)*(nColumn), level, fl_dp);
%         img_enc_s = func_SPIHT_Enc(I_W_S, max_bits_s, (nRow_s)*(nColumn_s), level_s, fl_dp);
%         min_k=min(img_enc(:));
%         max_k=max(img_enc(:));
%          %do hoffman coding
%          % Huffman table training
%         PMF = hist(img_enc_s(:),min_k:max_k) ; 
%         PMF = PMF/sum(PMF) ;
%         [ BinaryTree, HuffCode, BinCode, Codelengths] = buildHuffman( PMF );
%         bytestream = enc_huffman_new( img_enc(:)-min_k+1, BinCode, Codelengths);
%         len=len+length(bytestream);
%         %% Huffman decoding
%         qReconst_image = double(reshape( dec_huffman_new ( bytestream, BinaryTree, max(size(img_enc(:))) ), size(img_enc)))- 1 + min_k;
%         %% do decoding
%         img_dec = func_SPIHT_Dec(qReconst_image);
%         img_spiht = func_InvDWT(img_dec, S, Lo_R, Hi_R, level);
%         rec_image1(:,:,ij)=uint8(img_spiht);
%     end
% %use pretrained CNN do optimization
% load('pretrainedJPEGDnCNN.mat');
% prediction=denoiseImage(rec_image1(:,:,1),net);
% rec_image1 = cat(3,prediction,rec_image1(:,:,2:3));
% %performance
% rec_image_final=ycbcr2rgb(rec_image1);
% bitPerPixel(index)=(len*8)/(512*512);
% p(index)=psnr(rec_image_final,Orig_I1);
% disp(p(index))
% %%final reconstructed Image plot
% 
% % filename=sprintf('reconstruction_image_compression_ration%d.png',rate);
% % imwrite(rec_image_final, filename);
% % size(rec_image_final)
% % figure,
% % imshow(rec_image_final)
% end
% 
% %static result compare
%     load('Lena_static.mat');
%     figure( 'Name', 'rate-distortion' ) 
%     plot   (bitPerPixel, p, '-xb',staic_lena_dct(1,:),staic_lena_dct(2,:),'-xr');
%     xlabel( 'Bit per pixel' );
%     ylabel( 'PSNR [dB]' ) ;
%     axis([0.2 3.5 26 40]);
%     set(gca,'XTick',0.2:0.5:3.5);
%     set(gca,'YTick',26:1:40);
%    
%     title( 'R-D plot' ) ;
%     hold on ;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% modified
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% version
%color space transforms
%yy=rgb2ycbcr(Orig_I1);
Orig_I1 = double(Orig_I1);
[t_enc,t_off] = my_pca(Orig_I1); % pca version
yy = my_ictRGB2YCbCr(Orig_I1,t_enc,t_off);

Orig_I2 = yy;
%Orig_I2_small = rgb2ycbcr(image_small);
Orig_I2_small = my_ictRGB2YCbCr(image_small,t_enc,t_off);
%Note some useful Imformation
fl_dp=8;%it changes after PCA
[~,~,Layers]= size(Orig_I1);%checks the type of the oringinal Image, grey or color
index=0;
for rate=0.1:0.1:1.5%different compressions ration, like qScale
    index=index+1;
    len=0;
    for ij=1:3%do the compression each layer

        fprintf('plane %d computing\n\n',ij);

        Orig_I=Orig_I2(:,:,ij);
        Orig_I_small=Orig_I2_small(:,:,ij);
        %preparation for each maxbit of SPIHT Algorithm
        if ij==2
            [s_1,s_2] = size(Orig_I);
            s_1=ceil(s_1/2);
            s_2=ceil(s_2/2);

            [ss_1,ss_2] = size(Orig_I_small);
            ss_1=ceil(ss_1/2);
            ss_2=ceil(ss_2/2);
        elseif ij==3
            [s_1,s_2] = size(Orig_I);
            s_1=ceil(s_1/2);
            s_2=ceil(s_2/2);

            [ss_1,ss_2] = size(Orig_I_small);
            ss_1=ceil(ss_1/2);
            ss_2=ceil(ss_2/2);

        else
            [s_1,s_2] = size(Orig_I);
            [ss_1,ss_2] = size(Orig_I_small);
        end
        max_bits_s = floor(rate * (ss_1*ss_2));
        max_bits = floor(rate * (s_1*s_2));
        %OutSize = s_1;
        %image_spiht = zeros(ceil(size(Orig_I)/2));
        [nRow, nColumn] = size(Orig_I);
        [nRow_s, nColumn_s] = size(Orig_I_small);
         [n_1,n_2] = size(Orig_I);
         [n_1_s,n_2_s]=size(Orig_I_small);
        if n_1<n_2
            n_log = log2(n_2);
            n_log_s = log2(n_2_s);
        else
            n_log = log2(n_1);
            n_log_s = log2(n_1_s);
        end
        level =floor(n_log);
        level_s =floor(n_log_s);
        %% do dwt encoding
        type='bior4.4';%JEPG 2000
        [Lo_D,Hi_D,Lo_R,Hi_R] = wfilters(type);%??4 filters for dwt,this case haar transform

        [I_W, S] = func_DWT(Orig_I, level, Lo_D, Hi_D);
        [I_W_S,S1]=func_DWT(Orig_I_small,level_s, Lo_D, Hi_D);
        %I_W_3(:,:,ij)=uint8(I_W);
%         figure
%         imshow(I_W_3)
        % Use SPIHT to compress the bit stream
        img_enc = func_SPIHT_Enc(I_W, max_bits, (nRow)*(nColumn), level, fl_dp);
        img_enc_s = func_SPIHT_Enc(I_W_S, max_bits_s, (nRow_s)*(nColumn_s), level_s, fl_dp);
        min_k=min(img_enc(:));
        max_k=max(img_enc(:));
         %do hoffman coding
         % Huffman table training
        PMF = hist(img_enc_s(:),min_k:max_k) ; 
        PMF = PMF/sum(PMF) ;
        [ BinaryTree, HuffCode, BinCode, Codelengths] = buildHuffman( PMF );
        bytestream = enc_huffman_new( img_enc(:)-min_k+1, BinCode, Codelengths);
        len=len+length(bytestream);
        %% Huffman decoding
        qReconst_image = double(reshape( dec_huffman_new ( bytestream, BinaryTree, max(size(img_enc(:))) ), size(img_enc)))- 1 + min_k;
        %% do decoding
        img_dec = func_SPIHT_Dec(qReconst_image);
        img_spiht = func_InvDWT(img_dec, S, Lo_R, Hi_R, level);
        rec_image1(:,:,ij)=uint8(img_spiht);
    end
%use pretrained CNN do optimization
% load('pretrainedJPEGDnCNN.mat');
% prediction=denoiseImage(rec_image1(:,:,1),net);
% rec_image1 = cat(3,prediction,rec_image1(:,:,2:3));
%performance
rec_image1 = double(rec_image1);
rec_image_final = my_ictYCbCr2RGB(rec_image1,t_enc,t_off);
rec_image_final = deblock(rec_image_final);
%rec_image_final=ycbcr2rgb(rec_image1);
bitPerPixel(index)=(len*8)/(512*512);
%p(index)=psnr(rec_image_final,Orig_I1);
p(index)=calcPSNR(Orig_I1,rec_image_final);
disp(p(index))

%%final reconstructed Image plot

% filename=sprintf('reconstruction_image_compression_ration%d.png',rate);
% imwrite(rec_image_final, filename);
% size(rec_image_final)
% figure,
% imshow(rec_image_final)
end

%static result compare
    load('Lena_static.mat');
    figure( 'Name', 'rate-distortion' ) 
    plot   (bitPerPixel, p, '-xb',staic_lena_dct(1,:),staic_lena_dct(2,:),'-xr');
    xlabel( 'Bit per pixel' );
    ylabel( 'PSNR [dB]' ) ;
    axis([0.2 3.5 26 40]);
    set(gca,'XTick',0.2:0.5:3.5);
    set(gca,'YTick',26:1:40);
   
    title( 'R-D plot' ) ;
    hold on ;

    
function deblocked_img = deblock(image)
alpha = [0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	4	4	5	6	7	8	9	10	12	13 ...
      15	17	20	22	25	28	32	36	40	45	50	56	63	71	80	90	101	113	127	144	162	182	203	226	255	255];
beta = [0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	2	2	2	3	3	3	3	4	4	4 ...
     6	6	7	7	8	8	9	9	10	10	11	11	12	12	13	13	14	14	15	15	16	16	17	17	18	18];
alpha_fix = 10;
beta_fix = 9;
offset = 1; % adjust the value of the offset to adjust the filter leverage
weight = [0.1,0.4,0.4,0.1];
% sparsity_thres = 100; % adjust the value of the offset to pick "how" sparsity block to do the deblocking
% block_sparsity = load('foreman_sparsity.mat');
% block_sparsity = block_sparsity.block_nonzero;
[height_im, width_im, cdim_im] = size(image);
deblocked_img = image;
counter = 0;
block_length = 8;
%% Vertical
for hh = 1:3
for ii = block_length:block_length:height_im-block_length
    for jj = 1:block_length:width_im
        %sparsity_upper = block_sparsity(ii/block_length,(jj+block_length-1)/block_length);
        %sparsity_down = block_sparsity((ii+block_length)/block_length,(jj+block_length-1)/block_length);
        %if (sparsity_upper < sparsity_thres)&&(sparsity_down < sparsity_thres)
        upper_block = image(ii-block_length+1:ii,jj:jj+block_length-1,hh); % use luma value to decide
        down_block = image(ii+1:ii+block_length,jj:jj+block_length-1,hh);
        for kk = 1:block_length % in each block, compare the pixel on the boarder line
            upper_block_pixel = upper_block(block_length,kk);
            down_block_pixel = down_block(1,kk);
%             av_pixel = round(abs(upper_block_pixel - down_block_pixel)/2) + offset;
%             if av_pixel > 52
%                 av_pixel = 52;
%             end
            res1 = abs(upper_block_pixel - down_block_pixel);
            %res2 = abs(upper_block(block_length,kk) - upper_block(block_length-1,kk));
            %res3 = abs(down_block(1,kk) - down_block(2,kk));
            res2 = abs(down_block(2,kk) - upper_block(block_length-1,kk));
%             if (res1<alpha(av_pixel)) && (res2<beta(av_pixel)) && (res3<beta(av_pixel))
            if (res1<alpha_fix) && (res2<beta_fix)
                counter = counter + 1;
                block_ref = [image(ii-1,jj+kk-1,hh);image(ii,jj+kk-1,hh);image(ii+1,jj+kk-1,hh);image(ii+2,jj+kk-1,hh)];
                %block_ref = double(block_ref);
                weighted_sum_value = weight*block_ref;
                deblocked_img(ii,jj+kk-1,hh) = weighted_sum_value;
                deblocked_img(ii+1,jj+kk-1,hh) = weighted_sum_value;
            end
%         end
        end
    end
end
end
%% Horizontal
for hh = 1:3
for jj = block_length:block_length:width_im-block_length
    for ii = 1:block_length:height_im
%         sparsity_left = block_sparsity((ii+block_length-1)/block_length,jj/block_length);
%         sparsity_right = block_sparsity((ii+block_length-1)/block_length,(jj+block_length)/block_length);
%         if (sparsity_left < sparsity_thres)&&(sparsity_right < sparsity_thres)
        left_block = image(ii:ii+block_length-1,jj-block_length+1:jj,hh); % use luma value to decide
        right_block = image(ii:ii+block_length-1,jj+1:jj+block_length,hh);
        for kk = 1:block_length % in each block, compare the pixel on the boarder line
            left_block_pixel = left_block(kk,block_length);
            right_block_pixel = right_block(kk,1);
%             av_pixel = round(abs(left_block_pixel - right_block_pixel)/2) + offset;
%             if av_pixel > 52
%                 av_pixel = 52;
%             end
            res1 = abs(left_block_pixel - right_block_pixel);
            %res2 = abs(left_block(kk,block_length) - left_block(kk,block_length-1));
            %res3 = abs(right_block(kk,1) - right_block(kk,2));
            res2 = abs(right_block(kk,2) - left_block(kk,block_length-1));
%             if (res1<alpha(av_pixel)) && (res2<beta(av_pixel)) && (res3<beta(av_pixel))
            if (res1<alpha_fix) && (res2<beta_fix)
                counter = counter + 1;
                block_ref = [image(ii+kk-1,jj-1,hh);image(ii+kk-1,jj,hh);image(ii+kk-1,jj+1,hh);image(ii+kk-1,jj+2,hh)];
                %block_ref = double(block_ref);
                weighted_sum_value = weight*block_ref;
                deblocked_img(ii+kk-1,jj,hh) = weighted_sum_value;
                deblocked_img(ii+kk-1,jj+1,hh) = weighted_sum_value;
            end
%         end
        end
    end
end
end
disp(counter)
end

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

function rgb = my_ictYCbCr2RGB(yuv,t_enc,t_off)
t_inv = inv(t_enc);
yuv(:,:,1) = yuv(:,:,1) - t_off(1,:);
yuv(:,:,2) = yuv(:,:,2) - t_off(2,:);
yuv(:,:,3) = yuv(:,:,3) - t_off(3,:);
rgb(:,:,1) = t_inv(1,1)*yuv(:,:,1) + t_inv(1,2)*yuv(:,:,2) + t_inv(1,3)*yuv(:,:,3);
rgb(:,:,2) = t_inv(2,1)*yuv(:,:,1) + t_inv(2,2)*yuv(:,:,2) + t_inv(2,3)*yuv(:,:,3);
rgb(:,:,3) = t_inv(3,1)*yuv(:,:,1) + t_inv(3,2)*yuv(:,:,2) + t_inv(3,3)*yuv(:,:,3);
end

function yuv = my_ictRGB2YCbCr(rgb,t_enc,t_off)
yuv(:,:,1) = t_enc(1,1)*rgb(:,:,1) + t_enc(1,2)*rgb(:,:,2) + t_enc(1,3)*rgb(:,:,3) + t_off(1,:);
yuv(:,:,2) = t_enc(2,1)*rgb(:,:,1) + t_enc(2,2)*rgb(:,:,2) + t_enc(2,3)*rgb(:,:,3) + t_off(2,:);
yuv(:,:,3) = t_enc(3,1)*rgb(:,:,1) + t_enc(3,2)*rgb(:,:,2) + t_enc(3,3)*rgb(:,:,3) + t_off(3,:);
end

function PSNR = calcPSNR(Image, recImage)
mse = calcMSE(Image, recImage);
PSNR = 20*log10(255/sqrt(mse));
end

function MSE = calcMSE(Image, recImage)
[height, width, cdim] = size(Image);
MSE = sum(sum((double(Image) - double(recImage)).^2))/(height*width*cdim);
MSE = sum(MSE(:));
end