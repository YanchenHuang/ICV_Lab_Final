function [img_enc,S,level,Lo_D,Hi_D,Lo_R,Hi_R]=intraEncoder_1(img,rate,ij)
%input: img: YCbCr image of oringinal Image
%       rate: qScale
fl_dp=8;
%output: img_enc:encoded coeff of img,
%        S: index table of coeffs 
%        Lo_D,Hi_D,Lo_R,Hi_R: used filter of DWT
        Orig_I=img;
        %Orig_I_small=lena_small(:,:,ij);
        %preparation for each maxbit of SPIHT Algorithm
        if ij==2
            [s_1,s_2] = size(Orig_I);
            s_1=ceil(s_1/2);
            s_2=ceil(s_2/2);

%             [ss_1,ss_2] = size(Orig_I_small);
%             ss_1=ceil(ss_1/2);
%             ss_2=ceil(ss_2/2);
        elseif ij==3
            [s_1,s_2] = size(Orig_I);
            s_1=ceil(s_1/2);
            s_2=ceil(s_2/2);

%             [ss_1,ss_2] = size(Orig_I_small);
%             ss_1=ceil(ss_1/2);
%             ss_2=ceil(ss_2/2);

        else
            [s_1,s_2] = size(Orig_I);
%             [ss_1,ss_2] = size(Orig_I_small);
        end
%         max_bits_s = floor(rate * (ss_1*ss_2));
        max_bits = floor(rate * (s_1*s_2));
        %OutSize = s_1;
        %image_spiht = zeros(ceil(size(Orig_I)/2));
        [nRow, nColumn] = size(Orig_I);
%         [nRow_s, nColumn_s] = size(Orig_I_small);
         [n_1,n_2] = size(Orig_I);
%          [n_1_s,n_2_s]=size(Orig_I_small);
        if n_1<n_2
            n_log = log2(n_2);
%             n_log_s = log2(n_2_s);
        else
            n_log = log2(n_1);
%             n_log_s = log2(n_1_s);
        end
        level =floor(n_log);
%         level_s =floor(n_log_s);
        %% do dwt encoding
        type='bior4.4';%JEPG 2000
        [Lo_D,Hi_D,Lo_R,Hi_R] = wfilters(type);%??4 filters for dwt,this case haar transform

        [I_W, S] = func_DWT(Orig_I, level, Lo_D, Hi_D);
%         [I_W_S,S1]=func_DWT(Orig_I_small,level_s, Lo_D, Hi_D);
        
        % Use SPIHT to compress the bit stream
        img_enc = func_SPIHT_Enc(I_W, max_bits, (nRow)*(nColumn), level, fl_dp);
%         img_enc_s(:,:,ij) = func_SPIHT_Enc(I_W_S, max_bits_s, (nRow_s)*(nColumn_s), level_s, fl_dp);

end