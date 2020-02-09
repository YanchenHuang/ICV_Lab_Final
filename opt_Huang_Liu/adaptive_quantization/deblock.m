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
load('pretrainedJPEGDnCNN.mat');
prediction=denoiseImage(deblocked_img(:,:,1),net);
rec_image = cat(3,prediction,deblocked_img(:,:,2:3));
disp(counter)
end