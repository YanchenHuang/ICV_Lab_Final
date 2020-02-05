lena_small_rgb = double(imread('/Users/zoe/desktop/images/lena_small.tif'));
first_frame_rgb = double(imread('/Users/zoe/desktop/Sequences/foreman20_40_RGB/foreman0020.bmp'));
second_frame_rgb = double(imread('/Users/zoe/desktop/Sequences/foreman20_40_RGB/foreman0021.bmp'));

[t_enc,t_off] = my_pca(first_frame_rgb);

% lena_small = ictRGB2YCbCr(lena_small_rgb);
% first_frame = ictRGB2YCbCr(first_frame_rgb);
% second_frame = ictRGB2YCbCr(second_frame_rgb);
lena_small = my_ictRGB2YCbCr(lena_small_rgb,t_enc,t_off);
first_frame = my_ictRGB2YCbCr(first_frame_rgb,t_enc,t_off);
second_frame = my_ictRGB2YCbCr(second_frame_rgb,t_enc,t_off);

scales_1 = 0.15:0.3:1.5;
scales_2 = 1.8:1:6;
scales_3 = 7:2.5:15;
scales = [scales_1,scales_2,scales_3];
%scales = 1.0;
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%                     still image codec
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% for scaleIdx = 1 : numel(scales)
%     qScale   = scales(scaleIdx);
%     k_small  = IntraEncode(lena_small, qScale);
%     k        = IntraEncode(first_frame, qScale);
%         
%     %% use pmf of k_small to build and train huffman table
%     pmf_lena_small = hist(k_small(:),min(k):max(k));
%     pmf_lena_small = pmf_lena_small/sum(pmf_lena_small);
%     [BinaryTree, HuffCode, BinCode, Codelengths] = buildHuffman(pmf_lena_small);
%         
%     %% use trained table to encode k to get the bytestream
%     bytestream = enc_huffman_new(k-min(k)+1, BinCode, Codelengths);
%     bitPerPixel = (numel(bytestream)*8) / (numel(first_frame)/3);
%     
%     %% image reconstruction
%     k_rec = dec_huffman_new(bytestream,BinaryTree,max(size(k)))+min(k)-1;
%     I_rec = IntraDecode(k_rec, size(first_frame),qScale);
%     %I_rec_rgb = my_ictYCbCr2RGB(I_rec,t_enc,t_off);
%     I_rec_rgb = ictYCbCr2RGB(I_rec);
%     %[t_dec,p] = my_least_square(first_frame_rgb,I_rec,t_off);
%     %I_rec_rgb = my_ictYCbCr2RGB_ls(I_rec,t_dec,p,t_off);
%     deblock_I_rec_rgb = deblock(I_rec_rgb);
%     %deblock_I_rec_rgb = DeblockFilter(I_rec_rgb);
%     PSNR = calcPSNR(first_frame_rgb, I_rec_rgb);
%     
%     bitperpixel_av_still(scaleIdx) = bitPerPixel;
%     PSNR_av_still(scaleIdx) = PSNR;
% end

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%                     video codec
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for scaleIdx = 1 : numel(scales)
    %%%%%%%%%%%%%%%%% Encode and decode the first frame
    qScale   = scales(scaleIdx);
    k_small  = IntraEncode(lena_small, qScale);
    k        = IntraEncode(first_frame, qScale);
    
    %% train huffman table for intra-coded first frame on lena_small
    pmf_lena_small = hist(k_small(:),min(k):max(k));
    pmf_lena_small = pmf_lena_small/sum(pmf_lena_small);
    [BinaryTree, HuffCode, BinCode, Codelengths] = buildHuffman(pmf_lena_small);
    
    %% use trained table to encode k to get the bytestream
    bytestream = enc_huffman_new(k-min(k)+1, BinCode, Codelengths);
    bpp(1) = (numel(bytestream)*8) / (numel(first_frame)/3); % start to note down the bpp, this is for the 1st frame
    
    %% image reconstruction
    k_rec = dec_huffman_new(bytestream,BinaryTree,max(size(k)))+min(k)-1;
    I_rec = IntraDecode(k_rec, size(first_frame),qScale);
    %I_rec_rgb = ictYCbCr2RGB(I_rec);
    %I_rec_rgb = my_ictYCbCr2RGB(I_rec,t_enc,t_off);
    [t_dec,p] = my_least_square(first_frame_rgb,I_rec,t_off);
    I_rec_rgb = my_ictYCbCr2RGB_ls(I_rec,t_dec,p,t_off);
    I_rec_rgb = deblock(I_rec_rgb);
    %I_rec_rgb = DeblockFilter(I_rec_rgb);
    PSNR(1) = calcPSNR(first_frame_rgb, I_rec_rgb);

    %% train Huffman table for Motion vector on first Motion vector(between first and second frame)
    mv_indices = SSD(I_rec(:,:,1), second_frame(:,:,1)); % Note that here should be reconstucted img
    PMF = hist(mv_indices(:),1:81);
    PMF = PMF/sum(PMF);
    [BinaryTree_MV, HuffCode_MV, BinCode_MV, Codelengths_MV] = buildHuffman(PMF);
    
    %% train Huffman table for residuals on first residual(second frame)
    second_frame_rec = SSD_rec(I_rec,mv_indices);
    residual = second_frame - second_frame_rec;
    k = IntraEncode(residual, qScale); % according to fig.25, need to encode the residual before transmit
    PMF = hist(k(:),-1500:1500);
    PMF = PMF/sum(PMF);
    [BinaryTree_R, HuffCode_R, BinCode_R, Codelengths_R] = buildHuffman(PMF);
    
    fprintf("\tFrame %-4d, bitrate: %.4f [bps], PSNR: %.4f [dB]\n", 1, bpp(1), PSNR(1));
    %%%%%%%%%%%%%%%%% Encode and decode the 2 to N frame
    for i = 1:20
        current_frame_name = ['/Users/zoe/desktop/Sequences/foreman20_40_RGB/foreman00',int2str(20+i),'.bmp'];
        current_frame_rgb = double(imread(current_frame_name));
        %current_frame = ictRGB2YCbCr(current_frame_rgb); 
        current_frame = my_ictRGB2YCbCr(current_frame_rgb,t_enc,t_off);
        mv_indices = SSD(I_rec(:,:,1), current_frame(:,:,1));
        current_frame_rec = SSD_rec(I_rec,mv_indices);
        residual = current_frame - current_frame_rec;
        k = IntraEncode(residual, qScale);
        
        %% use trained table to encode MV to get the bytestream
        bytestream_MV = enc_huffman_new(mv_indices-1+1, BinCode_MV, Codelengths_MV);
        bpp_MV = (numel(bytestream_MV)*8) / (numel(current_frame)/3);
        bpp_MV_lc(i) = bpp_MV;
        
        %% use trained table to encode residual to get the bytestream
        bytestream_R = enc_huffman_new(k-(-1500)+1, BinCode_R, Codelengths_R);
        bpp_R = (numel(bytestream_R)*8) / (numel(current_frame)/3);
        bpp_R_lc(i) = bpp_R;
        
        %% note down the bpp to the array
        bpp(i+1) = bpp_MV + bpp_R;
        
        %% image reconstruction
        % mv_rec = dec_huffman_new(bytestream_MV, BinaryTree_MV,length(mv_indices(:)))+1)-1;
        % mv_rec = reshape(mv_rec,size(mv_indices));
        % since huffman is the lossless transmit, here mv_rec should be
        % exactly the same as mv_indices
        current_ssd_rec = SSD_rec(I_rec, mv_indices);
        
        % k_rec = dec_huffman_new(bytestream_R, BinaryTree_R,length(k(:)))+(-1500)-1;
        % since huffman is the lossless transmit, here k_rec should be the
        % same with k, thus the upper line could be commented
        residual_rec = IntraDecode(k, size(residual),qScale);
        
        I_rec = current_ssd_rec + residual_rec;
        %I_rec_rgb = ictYCbCr2RGB(I_rec);
        %I_rec_rgb = my_ictYCbCr2RGB(I_rec,t_enc,t_off);
        [t_dec,p] = my_least_square(current_frame_rgb,I_rec,t_off);
        I_rec_rgb = my_ictYCbCr2RGB_ls(I_rec,t_dec,p,t_off);
        I_rec_rgb = deblock(I_rec_rgb);
        %I_rec_rgb = DeblockFilter(I_rec_rgb);
        PSNR(i+1) = calcPSNR(current_frame_rgb,I_rec_rgb);
        fprintf("\tFrame %-4d, bitrate: %.4f [bps], PSNR: %.4f [dB]\n", i+1, bpp(i+1), PSNR(i+1));
    end
    bitperpixel_av(scaleIdx) = mean(bpp);
    PSNR_av(scaleIdx) = sum(PSNR)/numel(PSNR);
    fprintf("\tbitrate: %.4f [bps], PSNR: %.4f [dB]\n", bitperpixel_av(scaleIdx), PSNR_av(scaleIdx));
end

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%                          折线图
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
plot(bitperpixel_av_still,PSNR_av_still,'-*r',bitperpixel_av,PSNR_av,'-*b'); %线性，颜色，标记
%plot(bitperpixel_av_all(2,:),PSNR_av_all(2,:),'-*r',bitperpixel_av_all(3,:),PSNR_av_all(3,:),'-*b');
axis.XLim = [0 4.5];
axis.YLim = [24 42];
axis.XTick = 0:0.1:4.5;
axis.YTick = 24:2:42;
xlabel('bpp');  %x轴坐标描述
ylabel('PSNR[dB]'); %y轴坐标描述

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%                           sub-functions
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% put all used sub-functions here.
function imageYCbCr_dec = IntraDecode(image, img_size , qScale)
%  Function Name : IntraDecode.m
%  Input         : image (zero-run encoded image, 1xN)
%                  img_size (original image size)
%                  qScale(quantization scale)
%  Output        : dst   (decoded image)
% I_zigzag_dec = ZeroRunDec_EoB(image);
I_zigzag_dec = ZeroRunDec(image);
I_zigzag_dec = reshape(I_zigzag_dec,img_size(1)*8,(img_size(2)/64)*24);
I_quant_dec = blockproc(I_zigzag_dec, [64, 3], @(block_struct) DeZigZag8x8(block_struct.data));
I_dct_dec = blockproc(I_quant_dec, [8, 8], @(block_struct) DeQuant8x8(block_struct.data,qScale));
imageYCbCr_dec = blockproc(I_dct_dec, [8, 8], @(block_struct) IDCT8x8(block_struct.data));
end

function dst = IntraEncode(image, qScale)
%  Function Name : IntraEncode.m
%  Input         : image (Original RGB Image)
%                  qScale(quantization scale)
%  Output        : dst   (sequences after zero-run encoding, 1xN)
I_dct = blockproc(image, [8, 8], @(block_struct) DCT8x8(block_struct.data));
I_quant = blockproc(I_dct, [8, 8], @(block_struct) Quant8x8(block_struct.data,qScale));
I_zigzag = blockproc(I_quant, [8, 8], @(block_struct) ZigZag8x8(block_struct.data));
% dst = ZeroRunEnc_EoB(I_zigzag(:));
dst = ZeroRunEnc(I_zigzag(:));
end

%% and many more functions

function t_dec = find_best_dec(ori_rgbimg,ycbcr_img,t_enc,t_off,t_dec_ls)
rec_rgbimg = my_ictYCbCr2RGB(ycbcr_img,t_enc,t_off);
best_psnr = calcPSNR(ori_rgbimg, rec_rgbimg);
disp(best_psnr)
t_dec = inv(t_enc);
offset = 0.000001;
candidate_num = 1000;
scales = -candidate_num*offset:offset:candidate_num*offset;
for scaleIdx = 1 : numel(scales)
    qScale   = scales(scaleIdx);
    t_enc_mask = t_enc + qScale;
    rec_rgbimg = my_ictYCbCr2RGB(ycbcr_img,t_enc_mask,t_off);
    psnr_mask = calcPSNR(ori_rgbimg, rec_rgbimg);
    if psnr_mask > best_psnr
        t_dec = t_enc_mask;
    end
end
end

% function deblocked_img = deblock(image)
% alpha = [0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	4	4	5	6	7	8	9	10	12	13 ...
%       15	17	20	22	25	28	32	36	40	45	50	56	63	71	80	90	101	113	127	144	162	182	203	226	255	255];
% beta = [0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	2	2	2	3	3	3	3	4	4	4 ...
%      6	6	7	7	8	8	9	9	10	10	11	11	12	12	13	13	14	14	15	15	16	16	17	17	18	18];
% alpha_fix = 10;
% beta_fix = 9;
% offset = 1; % adjust the value of the offset to adjust the filter leverage
% weight = [0.1,0.4,0.4,0.1];
% % sparsity_thres = 100; % adjust the value of the offset to pick "how" sparsity block to do the deblocking
% % block_sparsity = load('foreman_sparsity.mat');
% % block_sparsity = block_sparsity.block_nonzero;
% [height_im, width_im, cdim_im] = size(image);
% deblocked_img = image;
% counter = 0;
% block_length = 16;
% %% Vertical
% for ii = block_length:block_length:height_im-block_length
%     for jj = 1:block_length:width_im
%         %sparsity_upper = block_sparsity(ii/block_length,(jj+block_length-1)/block_length);
%         %sparsity_down = block_sparsity((ii+block_length)/block_length,(jj+block_length-1)/block_length);
%         %if (sparsity_upper < sparsity_thres)&&(sparsity_down < sparsity_thres)
%         upper_block_luma = image(ii-block_length+1:ii,jj:jj+block_length-1,1); % use luma value to decide
%         down_block_luma = image(ii+1:ii+block_length,jj:jj+block_length-1,1);
%         for kk = 1:block_length % in each block, compare the pixel on the boarder line
%             upper_block_pixel = upper_block_luma(block_length,kk);
%             down_block_pixel = down_block_luma(1,kk);
% %             av_pixel = round(abs(upper_block_pixel - down_block_pixel)/2) + offset;
% %             if av_pixel > 52
% %                 av_pixel = 52;
% %             end
%             res1 = abs(upper_block_pixel - down_block_pixel);
%             res2 = abs(upper_block_luma(block_length,kk) - upper_block_luma(block_length-1,kk));
%             res3 = abs(down_block_luma(1,kk) - down_block_luma(2,kk));
% %             if (res1<alpha(av_pixel)) && (res2<beta(av_pixel)) && (res3<beta(av_pixel))
%             if (res1<alpha_fix) && (res2<beta_fix) && (res3<beta_fix)
%                 counter = counter + 1;
%                 for hh = 1:3
%                     block_ref = [image(ii-1,jj+kk-1,hh);image(ii,jj+kk-1,hh);image(ii+1,jj+kk-1,hh);image(ii+2,jj+kk-1,hh)];
%                     weighted_sum_value = weight*block_ref;
%                     deblocked_img(ii,jj+kk-1,hh) = weighted_sum_value;
%                     deblocked_img(ii+1,jj+kk-1,hh) = weighted_sum_value;
%                 end
%             end
% %         end
%         end
%     end
% end
% %% Horizontal
% for jj = block_length:block_length:width_im-block_length
%     for ii = 1:block_length:height_im
% %         sparsity_left = block_sparsity((ii+block_length-1)/block_length,jj/block_length);
% %         sparsity_right = block_sparsity((ii+block_length-1)/block_length,(jj+block_length)/block_length);
% %         if (sparsity_left < sparsity_thres)&&(sparsity_right < sparsity_thres)
%         left_block_luma = image(ii:ii+block_length-1,jj-block_length+1:jj,1); % use luma value to decide
%         right_block_luma = image(ii:ii+block_length-1,jj+1:jj+block_length,1);
%         for kk = 1:block_length % in each block, compare the pixel on the boarder line
%             left_block_pixel = left_block_luma(kk,block_length);
%             right_block_pixel = right_block_luma(kk,1);
% %             av_pixel = round(abs(left_block_pixel - right_block_pixel)/2) + offset;
% %             if av_pixel > 52
% %                 av_pixel = 52;
% %             end
%             res1 = abs(left_block_pixel - right_block_pixel);
%             res2 = abs(left_block_luma(kk,block_length) - left_block_luma(kk,block_length-1));
%             res3 = abs(right_block_luma(kk,1) - right_block_luma(kk,2));
% %             if (res1<alpha(av_pixel)) && (res2<beta(av_pixel)) && (res3<beta(av_pixel))
%             if (res1<alpha_fix) && (res2<beta_fix) && (res3<beta_fix)
%                 counter = counter + 1;
%                 for hh = 1:3
%                     block_ref = [image(ii+kk-1,jj-1,hh);image(ii+kk-1,jj,hh);image(ii+kk-1,jj+1,hh);image(ii+kk-1,jj+2,hh)];
%                     weighted_sum_value = weight*block_ref;
%                     deblocked_img(ii+kk-1,jj,hh) = weighted_sum_value;
%                     deblocked_img(ii+kk-1,jj+1,hh) = weighted_sum_value;
%                 end
%             end
% %         end
%         end
%     end
% end
% disp(counter)
% end

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
            %if (res1<alpha_fix) && (res2<beta_fix) && (res3<beta_fix)
            if (res1<alpha_fix) && (res2<beta_fix)
                counter = counter + 1;
                block_ref = [image(ii-1,jj+kk-1,hh);image(ii,jj+kk-1,hh);image(ii+1,jj+kk-1,hh);image(ii+2,jj+kk-1,hh)];
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
            %if (res1<alpha_fix) && (res2<beta_fix) && (res3<beta_fix)
            if (res1<alpha_fix) && (res2<beta_fix)
                counter = counter + 1;
                block_ref = [image(ii+kk-1,jj-1,hh);image(ii+kk-1,jj,hh);image(ii+kk-1,jj+1,hh);image(ii+kk-1,jj+2,hh)];
                weighted_sum_value = weight*block_ref;
                deblocked_img(ii+kk-1,jj,hh) = weighted_sum_value;
                deblocked_img(ii+kk-1,jj+1,hh) = weighted_sum_value;
            end
%         end
        end
    end
end
end
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

function [t_dec,p] = my_least_square(ori_img,new_img,t_off)
[height,width,channel] = size(ori_img);
red_seq = reshape(ori_img(:,:,1),[height*width,1]);
green_seq = reshape(ori_img(:,:,2),[height*width,1]);
blue_seq = reshape(ori_img(:,:,3),[height*width,1]);
% minus offset value
new_img(:,:,1) = new_img(:,:,1) - t_off(1,:);
new_img(:,:,2) = new_img(:,:,2) - t_off(2,:);
new_img(:,:,3) = new_img(:,:,3) - t_off(3,:);
new_img_mask = new_img;
new_img_mask = reshape(new_img_mask,[height*width,channel]);
%mask = inv(new_img.'*new_img)*(new_img.');
% red_coeff = mask*red_seq;
% green_coeff = mask*green_seq;
% blue_coeff = mask*blue_seq;
% use least square to get the inverse matrix
red_coeff = regress(red_seq,new_img_mask);
green_coeff = regress(green_seq,new_img_mask);
blue_coeff = regress(blue_seq,new_img_mask);
t_dec(1,:) = red_coeff;
t_dec(2,:) = green_coeff;
t_dec(3,:) = blue_coeff;
ori_img_seq = ori_img;
ori_img_seq = reshape(ori_img_seq,[height*width*channel,1]);
% convert ycbcr to RGB
rgb(:,:,1) = t_dec(1,1)*new_img(:,:,1) + t_dec(1,2)*new_img(:,:,2) + t_dec(1,3)*new_img(:,:,3);
rgb(:,:,2) = t_dec(2,1)*new_img(:,:,1) + t_dec(2,2)*new_img(:,:,2) + t_dec(2,3)*new_img(:,:,3);
rgb(:,:,3) = t_dec(3,1)*new_img(:,:,1) + t_dec(3,2)*new_img(:,:,2) + t_dec(3,3)*new_img(:,:,3);
rgb = reshape(rgb,[height*width*channel,1]);
% use polyfit to optimize the coefficient
p = polyfit(rgb,ori_img_seq,3);
end

function new_rec_rgb = my_ictYCbCr2RGB_ls(yuv,t_dec,p,t_off)
[height,width,channel] = size(yuv);
yuv(:,:,1) = yuv(:,:,1) - t_off(1,:);
yuv(:,:,2) = yuv(:,:,2) - t_off(2,:);
yuv(:,:,3) = yuv(:,:,3) - t_off(3,:);
rgb(:,:,1) = t_dec(1,1)*yuv(:,:,1) + t_dec(1,2)*yuv(:,:,2) + t_dec(1,3)*yuv(:,:,3);
rgb(:,:,2) = t_dec(2,1)*yuv(:,:,1) + t_dec(2,2)*yuv(:,:,2) + t_dec(2,3)*yuv(:,:,3);
rgb(:,:,3) = t_dec(3,1)*yuv(:,:,1) + t_dec(3,2)*yuv(:,:,2) + t_dec(3,3)*yuv(:,:,3);
rgb = reshape(rgb,[height*width*channel,1]);
new_rec_rgb = p(1)*(rgb.^3)+p(2)*(rgb.^2)+p(3)*rgb+p(4);
new_rec_rgb = reshape(new_rec_rgb,[height,width,channel]);
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

function rgb = ictYCbCr2RGB(yuv)
rgb(:,:,1) = yuv(:,:,1) + 1.402*yuv(:,:,3);
rgb(:,:,2) = yuv(:,:,1) - 0.344*yuv(:,:,2) - 0.714*yuv(:,:,3);
rgb(:,:,3) = yuv(:,:,1) + 1.772*yuv(:,:,2);
end

function yuv = ictRGB2YCbCr(rgb)
yuv(:,:,1) = 0.299*rgb(:,:,1) + 0.587*rgb(:,:,2) + 0.114*rgb(:,:,3);
yuv(:,:,2) = -0.169*rgb(:,:,1) - 0.331*rgb(:,:,2) + 0.5*rgb(:,:,3);
yuv(:,:,3) = 0.5*rgb(:,:,1) - 0.419*rgb(:,:,2) - 0.081*rgb(:,:,3);
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

function coeff = DCT8x8(block)
%  Input         : block    (Original Image block, 8x8x3)
%
%  Output        : coeff    (DCT coefficients after transformation, 8x8x3)
for i=1:3
    coeff(:,:,i) = dct2(block(:,:,i));
end
end

function block = IDCT8x8(coeff)
%  Function Name : IDCT8x8.m
%  Input         : coeff (DCT Coefficients) 8*8*3
%  Output        : block (original image block) 8*8*3
for i=1:3
    block(:,:,i) = idct2(coeff(:,:,i));
end
end

function quant = Quant8x8(dct_block, qScale)
%  Input         : dct_block (Original Coefficients, 8x8x3)
%                  qScale (Quantization Parameter, scalar)
%
%  Output        : quant (Quantized Coefficients, 8x8x3)
L = [16,11,10,16,24,40,51,61;
     12,12,14,19,26,58,60,55;
     14,13,16,24,40,57,69,56;
     14,17,22,29,51,87,80,62;
     18,55,37,56,68,109,103,77;
     24,35,55,64,81,104,113,92;
     49,64,78,87,103,121,120,101;
     72,92,95,98,112,100,103,99];
C = [17,18,24,47,99,99,99,99;
     18,21,26,66,99,99,99,99;
     24,13,56,99,99,99,99,99;
     47,66,99,99,99,99,99,99;
     99,99,99,99,99,99,99,99;
     99,99,99,99,99,99,99,99;
     99,99,99,99,99,99,99,99;
     99,99,99,99,99,99,99,99];

quant(:,:,1) = round(dct_block(:,:,1)./(L*qScale));
for i=2:3
quant(:,:,i) = round(dct_block(:,:,i)./(C*qScale));
end
end

function dct_block = DeQuant8x8(quant_block, qScale)
%  Function Name : DeQuant8x8.m
%  Input         : quant_block  (Quantized Block, 8x8x3)
%                  qScale       (Quantization Parameter, scalar)
%
%  Output        : dct_block    (Dequantized DCT coefficients, 8x8x3)
L = [16,11,10,16,24,40,51,61;
     12,12,14,19,26,58,60,55;
     14,13,16,24,40,57,69,56;
     14,17,22,29,51,87,80,62;
     18,55,37,56,68,109,103,77;
     24,35,55,64,81,104,113,92;
     49,64,78,87,103,121,120,101;
     72,92,95,98,112,100,103,99];
C = [17,18,24,47,99,99,99,99;
     18,21,26,66,99,99,99,99;
     24,13,56,99,99,99,99,99;
     47,66,99,99,99,99,99,99;
     99,99,99,99,99,99,99,99;
     99,99,99,99,99,99,99,99;
     99,99,99,99,99,99,99,99;
     99,99,99,99,99,99,99,99];

dct_block(:,:,1) = round(quant_block(:,:,1).*(L*qScale));
for i=2:3
dct_block(:,:,i) = round(quant_block(:,:,i).*(C*qScale));
end
end

function zz = ZigZag8x8(quant)
%  Input         : quant (Quantized Coefficients, 8x8x3)
%
%  Output        : zz (zig-zag scaned Coefficients, 64x3)
zigzag = [1,2,6,7,15,16,28,29;
          3,5,8,14,17,27,30,43;
          4,9,13,18,26,31,42,44;
          10,12,19,25,32,41,45,54;
          11,20,24,33,40,46,53,55;
          21,23,34,39,47,52,56,61;
          22,35,38,48,51,57,60,62;
          36,37,49,50,58,59,63,64];
for i=1:3
    quant_layer = quant(:,:,i);
    zz(zigzag(:),i) = quant_layer(:);
end
end

function coeffs = DeZigZag8x8(zz)
%  Function Name : DeZigZag8x8.m
%  Input         : zz    (Coefficients in zig-zag order)
%
%  Output        : coeffs(DCT coefficients in original order)
zigzag = [1,2,6,7,15,16,28,29;
          3,5,8,14,17,27,30,43;
          4,9,13,18,26,31,42,44;
          10,12,19,25,32,41,45,54;
          11,20,24,33,40,46,53,55;
          21,23,34,39,47,52,56,61;
          22,35,38,48,51,57,60,62;
          36,37,49,50,58,59,63,64];
for i=1:3
    zz_layer = zz(:,i);
    mask = zz_layer(zigzag(:));
    coeffs(:,:,i) = reshape(mask,8,8);
end
end

function zze = ZeroRunEnc(zz)
%  Input         : zz (Zig-zag scanned block, 1xN)
k = 1;
i =1;
while i <= numel(zz)
    if zz(i) ~= 0
        zze(k) = zz(i);
        k = k+1;
        i = i+1;
    else
        zze(k) = 0;
        j = i;
        count = 0;
        while j<=numel(zz) && zz(j) == 0 && count<256
            count = count+1;
            j = j+1;
        end
        zze(k+1) = count-1;
        k = k+2;
        i = j;
    end
end
%  Output        : zze (zero-run-level encoded block, 1xM)
end

function dst = ZeroRunDec(src)
%  Function Name : ZeroRunDec.m zero run level decoder
%  Input         : src (zero run code)
k = 1;
i = 1;
dst = [];
while i <= numel(src)
    if src(i) ~= 0
        value = src(i);
        dst =[dst,value];
        k = k+1;
        i = i+1;
    else
        for j = 0:src(i+1)
            dst(k+j) = 0;
        end
        k = k+j+1;
        i = i+2;
    end
end
%  Output        : dst (reconstructed source)
end

function zze = ZeroRunEnc_EoB(zz)
%  Input         : zz (Zig-zag scanned sequence, 1xN)
%
%  Output        : zze (zero-run-level encoded sequence, 1xM)
EOB = 1000;
loop = ceil(length(zz)/64);
zze = [];
if loop==1
    zze = blockZeroRunEnc_EoB(zz,EOB);
else
    for i=1:loop-1
        an = blockZeroRunEnc_EoB(zz((i-1)*64+1:i*64),EOB);
        length_an = length(an);
        zze(end+1:end+length_an) = an;
    end
    an = blockZeroRunEnc_EoB(zz((loop-1)*64+1:end),EOB);
    length_an = length(an);
    zze(end+1:end+length_an) = an;
end
end

function zze = blockZeroRunEnc_EoB(zz, EOB)
%  Input         : zz (Zig-zag scanned block sequence, 1x64)
%                  EOB (End Of Block symbol, scalar)
%
%  Output        : zze (zero-run-level encoded sequence, 1xM)
zero_counter = 0;
zero_mode_flag = 0;
old_seq_counter = 1;
new_seq_counter = 1;
while old_seq_counter<=length(zz)
    if zz(old_seq_counter)~=0 && zero_mode_flag==0
        zze(new_seq_counter) = zz(old_seq_counter);
        new_seq_counter = new_seq_counter + 1;
        old_seq_counter = old_seq_counter + 1;
        if old_seq_counter==length(zz)+1
            break
        end
    end
    if zz(old_seq_counter)==0 && zero_mode_flag==0
        if old_seq_counter == length(zz)
            zze(new_seq_counter) = EOB;
            break
        end
        if zz(old_seq_counter+1)==0
            if old_seq_counter+1 == length(zz)
                zze(new_seq_counter) = EOB;
                break
            else
                zero_mode_flag = 1;
                old_seq_counter = old_seq_counter + 1;
                zze(new_seq_counter) = 0;
                new_seq_counter = new_seq_counter + 1;
            end
        else
            zze(new_seq_counter) = 0;
            zze(new_seq_counter+1) = 0;
            new_seq_counter = new_seq_counter + 2;
            old_seq_counter = old_seq_counter + 1;
        end
    end
    if zz(old_seq_counter)==0 && zero_mode_flag==1
        zero_counter = zero_counter + 1;
        if zz(old_seq_counter+1)==0
            if old_seq_counter+1 == length(zz)
                zze(new_seq_counter-1) = EOB;
                break 
            else
                old_seq_counter = old_seq_counter + 1;
            end
        else
            zero_mode_flag = 0;
            zze(new_seq_counter) = zero_counter;
            zero_counter = 0;
            new_seq_counter = new_seq_counter + 1;
            old_seq_counter = old_seq_counter + 1;
        end
    end
end
end

function dst = ZeroRunDec_EoB(src)
%  Function Name : ZeroRunDec1.m zero run level decoder
%  Input         : src (zero run encoded sequence 1xM with EoB signs)
%
%  Output        : dst (reconstructed zig-zag scanned sequence 1xN)
EoB = 1000;
old_seq_counter = 1;
new_seq_counter = 1;
one_block_number = 64;
num_block = 1;
while old_seq_counter<=length(src)
    if src(old_seq_counter)==EoB
        dst(new_seq_counter:one_block_number*num_block) = 0;
        new_seq_counter = one_block_number*num_block + 1;
        old_seq_counter = old_seq_counter + 1;
        num_block = num_block + 1;
        if old_seq_counter == length(src) + 1
            break
        end
    else
        if src(old_seq_counter)~=0
            dst(new_seq_counter) = src(old_seq_counter);
            new_seq_counter = new_seq_counter + 1;
            old_seq_counter = old_seq_counter + 1;
            if old_seq_counter == length(src) + 1
                break
            end
        end
        if src(old_seq_counter)==0
            zero_number = src(old_seq_counter+1);
            dst(new_seq_counter:new_seq_counter+zero_number) = 0;
            new_seq_counter = new_seq_counter+zero_number+1;
            old_seq_counter = old_seq_counter + 2;
        end
    end
end
end

function motion_vectors_indices = SSD(ref_image, image)
%  Input         : ref_image(Reference Image, size: height x width)
%                  image (Current Image, size: height x width)
%
%  Output        : motion_vectors_indices (Motion Vector Indices, size: (height/8) x (width/8) x 1 )
%% size(ref_image) = size(image) = (288, 352) = (36*8,44*8)
montion_vector_matrix = reshape((1:81),9,9)';
% in ref image, consider -+4 search range
ref_image = padarray(ref_image,[4 4],0);
% in ref image, for the (+4,+4) edge point, complete that 8*8 block with 0
ref_image = padarray(ref_image,[7 7],0,'post');
% get the height,width of the current image
[height,width] = size(image);
for i=1:8:height
    for j=1:8:width
        % for each 8*8 block in the current image, use that as a reference to find match in ref image
        current_block = image(i:i+7,j:j+7);
        best_SSE = 99999999;
        for y=i:i+8
            for x=j:j+8
                ref_block = ref_image(y:y+7,x:x+7);
                mask = (current_block - ref_block).^2;
                sum_sse = sum(sum(mask));
                if sum_sse < best_SSE
                    best_SSE = sum_sse;
                    best_x_index = x-j+1;
                    best_y_index = y-i+1;
                end
            end
        end
        motion_vectors_indices((i-1)/8+1,(j-1)/8+1) = montion_vector_matrix(best_y_index,best_x_index);
    end
end
end

function rec_image = SSD_rec(ref_image, motion_vectors)
%  Input         : ref_image(Reference Image, YCbCr image)
%                  motion_vectors
%
%  Output        : rec_image (Reconstructed current image, YCbCr image)
%% size(ref_image) = (288,352,3)
%% size(motion_vectors) = (36,44)
[height,width,~] = size(ref_image);
for i=1:8:width
    for j=1:8:height
        % change motion index into x,y montion index
        motion_vector_value = motion_vectors((j-1)/8+1,(i-1)/8+1);
        y = ceil(motion_vector_value/9);
        x = mod(motion_vector_value,9);
        if x == 0
            x = 9;
        end
        % in block (i:i+7,j:j+7), the first element (i,j) is the center of the motion matrix
        % use the x,y motion index to find offset/distance to the center of the motion matrix
        x_offset = x-5;
        y_offset = y-5;
        % give ref_image value to rec_image
        if (i+x_offset)>0 && (i+x_offset+7)<=width && (j+y_offset)>0 && ((j+y_offset+7))<=height
            rec_image(j:j+7,i:i+7,:) = ref_image(j+y_offset:j+y_offset+7,i+x_offset:i+x_offset+7,:);
        else
            rec_image(j:j+7,i:i+7,:) = ref_image(j:j+7,i:i+7,:);
        end
    end
end
end