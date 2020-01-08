%this is the main script of how to run the SPIHT and DWT algorithm
clc
clear all

FileName='Lena.tif';
Orig_I1=imread(FileName);
image_small=double(imread('lena_small.tif'));

%color space transforms
yy=rgb2ycbcr(Orig_I1);
Orig_I2 = yy;

%Note some useful Imformation
fl_dp=8;%it changes after PCA
[~,~,Layers]= size(Orig_I1);%checks the type of the oringinal Image, grey or color
rate=1.2;%bits per pixel
for ij=1:3%do the compression each layer
    
    Orig_I=Orig_I2(:,:,ij);
    [s_1,s_2] = size(Orig_I);
     s_1=ceil(s_1/2);
     s_2=ceil(s_2/2);
     max_bits = floor(rate * (s_1*s_2));
     [nRow, nColumn] = size(Orig_I);
     [n_1,n_2] = size(Orig_I);
    if n_1<n_2
        n_log = log2(n_2);
    else
        n_log = log2(n_1);
    end
    %% do dwt encoding
    level =floor(n_log);
    type='bior4.4';
    [Lo_D,Hi_D,Lo_R,Hi_R] = wfilters(type);%£¬4 filters for dwt,this case haar transform
    
    [I_W, S] = func_DWT(Orig_I, level, Lo_D, Hi_D);
    I_W_3(:,:,ij)=uint8(I_W);
    
    % Use SPIHT to compress the bit stream
    img_enc = func_SPIHT_Enc(I_W, max_bits, (nRow)*(nColumn), level, fl_dp);
    min_k=min(img_enc(:));
%      %do hoffman coding
%      % Huffman table training
%     PMF = hist(image_small(:),0:255) ; 
%     PMF = PMF/sum(PMF) ;
%     [ BinaryTree, HuffCode, BinCode, Codelengths] = buildHuffman( PMF );
%     bytestream = enc_huffman_new( img_enc(:)-min_k+1, BinCode, Codelengths);
%     %% Huffman decoding
%     qReconst_image = double(reshape( dec_huffman_new ( bytestream, BinaryTree, max(size(img_enc(:))) ), size(img_enc)))- 1 + min_k;
    %% do decoding

    img_dec = func_SPIHT_Dec(img_enc);
    img_spiht = func_InvDWT(img_dec, S, Lo_R, Hi_R, level);
    rec_image1(:,:,ij)=uint8(img_spiht);
end
%use pretrained CNN do optimization
load('pretrainedJPEGDnCNN.mat');
prediction=denoiseImage(rec_image1(:,:,1),net);
rec_image = cat(3,prediction,rec_image1(:,:,2:3));

%%plot
rec_image_final=ycbcr2rgb(rec_image);
filename=sprintf('reconstruction_image_compression_ration%d.png',rate);
imwrite(rec_image_final, filename);
size(rec_image_final)
figure,
imshow(rec_image_final)
p=psnr(rec_image_final,Orig_I1)