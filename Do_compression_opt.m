%this is the main script of how to run the SPIHT and DWT algorithm
clc
clear all

FileName='Lena.tif';
Orig_I1=imread(FileName);
image_small=double(imread('lena_small.tif'));

%color space transforms
 yy=rgb2ycbcr(Orig_I1);
%[t_enc,t_off] = my_pca(Orig_I1); % pca version:

Orig_I2 = yy;
Orig_I2_small = rgb2ycbcr(image_small);
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
        I_W_3(:,:,ij)=uint8(I_W);
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
load('pretrainedJPEGDnCNN.mat');
prediction=denoiseImage(rec_image1(:,:,1),net);
rec_image1 = cat(3,prediction,rec_image1(:,:,2:3));
%performance
rec_image_final=ycbcr2rgb(rec_image1);
bitPerPixel(index)=(len*8)/(512*512);
p(index)=psnr(rec_image_final,Orig_I1);

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
