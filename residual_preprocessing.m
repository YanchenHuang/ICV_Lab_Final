clc
clear all


lena_small_rgb = imread('lena_small.tif');
first_frame_rgb = (imread('D:\我的坚果云\LAB_File\1\5\Chapter5\foreman20_40_RGB\foreman0020.bmp'));
%first_frame_rgb=imread("Lena.tif");
second_frame_rgb = double(imread('D:\我的坚果云\LAB_File\1\5\Chapter5\foreman20_40_RGB\foreman0021.bmp'));
lena_small = ictRGB2YCbCr(lena_small_rgb);
first_frame = ictRGB2YCbCr(first_frame_rgb);
second_frame = ictRGB2YCbCr(second_frame_rgb);

% scales_1 = 0.15:0.3:1.5;
% scales_2 = 1.8:1:6;
scales_3 = 0.1:0.1:1.0;
scales = [scales_3];
qScalar=1;
%ratial=[1:1:10];

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%                     still image codec
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for scaleIdx = 1 : numel(scales)
   for ij=1:3
    lena_small_1=lena_small(:,:,ij);
    first_frame_1=first_frame(:,:,ij);
    qScale   = scales(scaleIdx);
    [k_small,~,~,~,~,~,~]  = intraEncoder_1(lena_small_1, qScale,ij,qScalar);
    [k,S,level,Lo_D,Hi_D,Lo_R,Hi_R]        = intraEncoder_1(first_frame_1, qScale,ij,qScalar);

    %% use pmf of k_small to build and train huffman table
    pmf_lena_small = hist(k_small(:),min(k(:)):max(k(:)));
    pmf_lena_small = pmf_lena_small/sum(pmf_lena_small);
    [BinaryTree, HuffCode, BinCode, Codelengths] = buildHuffman(pmf_lena_small);

    %% use trained table to encode k to get the bytestream
    bytestream = enc_huffman_new(k-min(k)+1, BinCode, Codelengths);
    bitPerPixel(ij) = (numel(bytestream)*8) / (numel(first_frame)/3);

    %% image reconstruction;
     k_rec=double(reshape( dec_huffman_new ( bytestream, BinaryTree, max(size(k(:))) ), size(k)))- 1 + min(k(:));
     I_rec(:,:,ij) = intraDecoder_1(k,S, Lo_R, Hi_R, level,qScalar);
   end
    %I_rec_rgb=ictYCbCr2RGB(I_rec);
    %first_frame_RGB=ictYCbCr2RGB(first_frame);
    PSNR = calcPSNR(first_frame,I_rec);

    bitperpixel_av_still(scaleIdx) = sum(bitPerPixel);
    PSNR_av_still(scaleIdx) = PSNR;
end

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%                     video code
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for scaleIdx = 1 : numel(scales)
    for ji=1:3
        lena_small_1=lena_small(:,:,ji);
        first_frame_1=first_frame(:,:,ji);
        qScale   = scales(scaleIdx);
        %qScale=ratial(scaleIdx);
        [k_small,~,~,~,~,~,~]  = intraEncoder_1(lena_small_1, qScale,ji,qScalar);
        [k,S,level,Lo_D,Hi_D,Lo_R,Hi_R]        = intraEncoder_1(first_frame_1, qScale,ji,qScalar);
        
        %% use pmf of k_small to build and train huffman table
        pmf_lena_small = hist(k_small(:),min(k):max(k));
        pmf_lena_small = pmf_lena_small/sum(pmf_lena_small);
        [BinaryTree, HuffCode, BinCode, Codelengths] = buildHuffman(pmf_lena_small);
        
       %% use trained table to encode k to get the bytestream
        bytestream = enc_huffman_new(k-min(k)+1, BinCode, Codelengths);
        bpp_1(ji) = (numel(bytestream)*8) / (numel(first_frame)/3);
        
       %% image reconstruction
        k_rec = dec_huffman_new(bytestream,BinaryTree,max(size(k)))+min(k)-1;
        I_rec(:,:,ji) = intraDecoder_1(k_rec,S, Lo_R, Hi_R, level,qScalar);
    end
    I_rec_rgb_still=ictYCbCr2RGB(I_rec);
    bpp(1)=sum(bpp_1(:));
    I_rec_rgb = double(ycbcr2rgb(I_rec));
    PSNR(1) = calcPSNR(first_frame_rgb, I_rec_rgb);
    I_rec=double(I_rec);
    
    %% train Huffman table for Motion vector on first Motion vector(between first and second frame)    
    mv_indices = SSD(I_rec(:,:,1), second_frame(:,:,1)); % Note that here should be reconstucted img
    PMF = hist(mv_indices(:),1:81);
    PMF = PMF/sum(PMF);
    [BinaryTree_MV, HuffCode_MV, BinCode_MV, Codelengths_MV] = buildHuffman(PMF);
    
    %% train Huffman table for residuals on first residual(second frame)
    second_frame_rec = SSD_rec(I_rec,mv_indices);
    residual = second_frame - second_frame_rec;
    %preprocessing of residual image
    %residual=residual*10;
    for L=1:3
        k = intraEncoder_1(residual(:,:,L), qScale,L,qScalar); %
        PMF = hist(k(:),-1500:1500);
        PMF = PMF/sum(PMF);
        if L==1
            [BinaryTree_R_1, HuffCode_R_1, BinCode_R_1, Codelengths_R_1] = buildHuffman(PMF);
        elseif L==2
            [BinaryTree_R_2, HuffCode_R_2, BinCode_R_2, Codelengths_R_2] = buildHuffman(PMF);
        else
            [BinaryTree_R_3, HuffCode_R_3, BinCode_R_3, Codelengths_R_3] = buildHuffman(PMF);
        end
    end
        %%%%%%%%%%%%%%%% Encode and decode the 2 to N frame
        fprintf("video coding begin\n");
        for i = 2:20
            fprintf("frame____ %d\n_",i);
            current_frame_name = ['D:\我的坚果云\LAB_File\1\5\Chapter5\foreman20_40_RGB\foreman00',int2str(20+i),'.bmp'];
            current_frame_rgb = double(imread(current_frame_name));
            current_frame = ictRGB2YCbCr(current_frame_rgb);
            mv_indices = SSD(I_rec(:,:,1), current_frame(:,:,1));
            current_frame_rec = SSD_rec(I_rec,mv_indices);
            %preprocess of residual signal
            residual_1 = current_frame - current_frame_rec;
            residual_1_min=min(min(residual_1(:)));
            residual_1=residual_1-residual_1_min;
            %min(min(residual_1(:)))
            %test_frame=double(uint8(current_frame_rec)+uint8(residual_1));
            
            %residual=residual*10;
            for Layer=1:3
                %fprintf("Layer %d\n",Layer);
                k = intraEncoder_1(residual_1(:,:,Layer),qScale,Layer,qScalar);
                
                %% use trained table to encode MV to get the bytestream
                bytestream_MV = enc_huffman_new(mv_indices-1+1, BinCode_MV, Codelengths_MV);
                bpp_MV_1 = (numel(bytestream_MV)*8) / (numel(current_frame)/3);
                
                %% use trained table to encode residual to get the bytestream
                if Layer==1
                    bytestream_R = enc_huffman_new(k-(-1500)+1, BinCode_R_1, Codelengths_R_1);
                    bpp_R(Layer) = (numel(bytestream_R)*8) / (numel(current_frame)/3);
                elseif Layer==2
                    bytestream_R = enc_huffman_new(k-(-1500)+1, BinCode_R_2, Codelengths_R_2);
                    bpp_R(Layer) = (numel(bytestream_R)*8) / (numel(current_frame)/3);
                else
                    bytestream_R = enc_huffman_new(k-(-1500)+1, BinCode_R_1, Codelengths_R_1);
                    bpp_R(Layer) = (numel(bytestream_R)*8) / (numel(current_frame)/3);
                end
            
            %% note down the bpp to the array
            bpp(i) = bpp_MV_1 + sum(bpp_R(:));
            
            %% image reconstruction
            % mv_rec = dec_huffman_new(bytestream_MV, BinaryTree_MV,length(mv_indices(:)))+1)-1;
            % mv_rec = reshape(mv_rec,size(mv_indices));
            % since huffman is the lossless transmit, here mv_rec should be
            % exactly the same as mv_indices
            current_ssd_rec = SSD_rec(I_rec, mv_indices);
            
            % k_rec = dec_huffman_new(bytestream_R, BinaryTree_R,length(k(:)))+(-1500)-1;
            % since huffman is the lossless transmit, here k_rec should be the
            % same with k, thus the upper line could be commented
            residual_rec_1 = intraDecoder_1(k,S,Lo_R,Hi_R,level,qScalar);
            %residua_rec=residual_rec/10;
            I_rec(:,:,Layer) = current_ssd_rec(:,:,Layer) + residual_rec_1+residual_1_min;
            end
            I_rec_rgb=ictYCbCr2RGB(I_rec);
            PSNR(i) = calcPSNR(current_frame_rgb,I_rec_rgb);
        end
        bitperpixel_av(scaleIdx) = mean(bpp);
        PSNR_av(scaleIdx) = sum(PSNR)/numel(PSNR);
end

    
    % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%                          鎶樼嚎鍥?
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    plot(bitperpixel_av_still,PSNR_av_still,'-*r',bitperpixel_av,PSNR_av,'-*b'); %绾挎?э紝棰滆壊锛屾爣璁?
    axis.XLim = [0 4.5];
    axis.YLim = [24 42];
    axis.XTick = 0:0.1:4.5;
    axis.YTick = 24:2:42;
    xlabel('bpp');  %x杞村潗鏍囨弿杩?
    ylabel('PSNR[dB]'); %y杞村潗鏍囨弿杩?
    
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

