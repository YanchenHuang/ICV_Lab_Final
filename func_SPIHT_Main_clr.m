function [outfilename,bpp,MSE,psnr,compr] = func_SPIHT_Main_clr(infilename,rate,tt,im_info,im_name)
% Matlab implementation of SPIHT
%
% Main function
%
% input:    Orig_I : the original image.
%           rate : bits per pixel
% output:   img_spiht
%
multiWaitbar( 'CloseAll' );
tic
fprintf('-----------   Welcome to SPIHT Matlab    ----------------\n');
fprintf('-------DWT------------SPIHT----------HUFFMAN-----------\n');
fprintf('-----------   Load Image   ----------------\n');

%color images

%infilename = 'test9.bmp';
%outfilename = 'test9_reconstruct.bmp';

len=0;
%Orig_I1 = (imread(infilename));
Orig_I1 = (infilename);

yy=rgb2ycbcr(Orig_I1);
Orig_I2 = double(yy);
fl_sz=im_info.FileSize;
fl_dp=im_info.BitDepth;
fprintf('done!\n\n');

multiWaitbar( 'Image Encoding...', 1/5, 'Color', [0.4 0.1 0.5] );

for ij=1:3
    
    fprintf('plane %d computing\n\n',ij);
    
    Orig_I=Orig_I2(:,:,ij);
    
    %rate = 1;
    if ij==2
        [s_1,s_2] = size(Orig_I);
        s_1=ceil(s_1/2);
        s_2=ceil(s_2/2);
    elseif ij==3
        [s_1,s_2] = size(Orig_I);
        s_1=ceil(s_1/2);
        s_2=ceil(s_2/2);
    else
        [s_1,s_2] = size(Orig_I);
    end
    
    max_bits = floor(rate * (s_1*s_2));
    OutSize = s_1;
    image_spiht = zeros(ceil(size(Orig_I)/2));
    [nRow, nColumn] = size(Orig_I);
    
    multiWaitbar( 'Image Encoding...', (ij+1)/5, 'Color', [0.4 0.1 0.5] );
    
    fprintf('-----------   Wavelet Decomposition   ----------------\n');
    [n_1,n_2] = size(Orig_I);
    if n_1<n_2
        n_log = log2(n_2);
    else
        n_log = log2(n_1);
    end
    level =floor(n_log);
    % wavelet decomposition level can be defined by users manually.
    
    type = tt;   %'bior4.4'
    fprintf('filter type = ');
    disp(tt)
    [Lo_D,Hi_D,Lo_R,Hi_R] = wfilters(type);%使用现成函数产生，4个filter用于后续dwt
    
    [I_W, S] = func_DWT(Orig_I, level, Lo_D, Hi_D);
    
    I_W_3(:,:,ij)=uint8(I_W);
    
    fprintf('done!\n');
    
    fprintf('----------- SPIHT  Encoding   ----------------\n');
    img_enc = func_SPIHT_Enc(I_W, max_bits, (nRow)*(nColumn), level, fl_dp);
    fprintf('done!\n');
    
    fid = fopen('spenc.txt','w');
    fprintf(fid, '%d', img_enc);
    fclose(fid);
    
    fprintf('-----------  HUFFMAN encoding   ----------------\n');
    fhstartclr(img_enc,ij,rate,im_name);
    fprintf('done!\n');
    
end

multiWaitbar( 'Image Encoding...', 5/5, 'Color', [0.4 0.1 0.5] );
t1=toc;


if 1>fl_dp<=24
    image(uint8(I_W_3));
elseif 24>fl_dp<=48
    image(uint16(I_W_3));
else
    image(logical(I_W_3));
end
%image(uint8(I_W_3));   % for displaying the DWT coefficients.....
axis image;
pause

fprintf('\n\n\nthe encoding block generates compressed file and taken by decoding block\n\n\n\n');
tic

multiWaitbar( 'Image Decoding...', 1/5, 'Color', [0.4 0.1 0.5] );

for ij1=1:3
    
    xx2=num2str(rate);
    %nme=strcat(im_name,' rate- ',ax2,'.txt');
    
    if ij1 == 1
        xx3=num2str(ij1);
        nme=strcat(im_name,' rate- ',xx2,'-',xx3,'.txt');
    elseif ij1==2
        xx3=num2str(ij1);
        nme=strcat(im_name,' rate- ',xx2,'-',xx3,'.txt');
    else
        xx3=num2str(ij1);
        nme=strcat(im_name,' rate- ',xx2,'-',xx3,'.txt');
    end
    id1 = fopen(nme,'r');
    a11 = fscanf(id1,'%c',inf);
    fclose(id1);
    
    lngt =(length(a11));
    
    
    fprintf('plane %d computing\n\n',ij1);
    
    multiWaitbar( 'Image Decoding...', (ij1+1)/5, 'Color', [0.4 0.1 0.5] );
    
    fprintf('-----------  HUFFMAN Decoding   ----------------\n');
    huff_dec=fhdecode2clr(nme);
    fprintf('done!\n');
    
    dx_level=huff_dec(4);
    dx_BtDpt=huff_dec(5);
    
    fprintf('-----------  SPIHT Decoding   ----------------\n');
    img_dec = func_SPIHT_Dec(huff_dec);
    fprintf('done!\n');
    
    
    fprintf('-----------   Wavelet Reconstruction   ----------------\n');
    img_spiht = func_InvDWT(img_dec, S, Lo_R, Hi_R, dx_level);
    
    fprintf('done!\n');
    
    len=len+lngt;
    y2r(:,:,ij1)=uint8(img_spiht);
    
end
t2=toc;
origre=ycbcr2rgb(y2r);
%outfilename=uint8(origre);

if 1>dx_BtDpt<=24
    outfilename = uint8(origre);
elseif 24>dx_BtDpt<=48
    outfilename = uint16(origre);
else
    outfilename = logical(origre);
end

%imwrite((origre), gray(256), outfilename, 'bmp');


fprintf('\n\n-----------   Performance   ----------------\n');

[sz1,sz2,sz3]=size(Orig_I1);
fprintf('Size of image is %dx%d \n',sz1,sz2);
compr1=(len/(fl_sz));
bpp=(compr1*8);
compr=(compr1*100);
fprintf('Compression Ratio = %.3f percent (%.2f : 1)  \n',(compr1*100),(1/compr1));
fprintf('The bitrate is %.2f bpp (with rate %.2f in the encoding)\n',bpp ,rate);

jn1=double(origre);
jn2=double(Orig_I1);
[MSE,snr,psnr]=Peak_SNR(jn1,jn2);
%dx_bit=dx_BtDpt/3;
%Q=max(Orig_I1(:));
%Q = 255;
%MSE = sum(sum(sum((double(outfilename)-double(Orig_I1)).^2)))/(sz1*sz2*sz3);
fprintf('The MSE performance is %.2f \n', MSE);
%psnr=10*log10(Q*Q/MSE);
fprintf('The psnr performance is %.2f dB\n', psnr);

fprintf('Time consumption \n');
fprintf('Encoding = %.3f sec\n',t1);
fprintf('Decoding = %.3f sec\n',t2);

multiWaitbar( 'Image Decoding...', 5/5, 'Color', [0.4 0.1 0.5] );
multiWaitbar( 'CloseAll' );

%[X1,map1]=imread(infilename);
%[X2,map2]=imread(outfilename);
%subplot(1,2,1), imshow(X1,map1)
%subplot(1,2,2), imshow(X2,map2)