function rgb = my_ictYCbCr2RGB(yuv,t_enc,t_off)
t_inv = inv(t_enc);
yuv(:,:,1) = yuv(:,:,1) - t_off(1,:);
yuv(:,:,2) = yuv(:,:,2) - t_off(2,:);
yuv(:,:,3) = yuv(:,:,3) - t_off(3,:);
rgb(:,:,1) = t_inv(1,1)*yuv(:,:,1) + t_inv(1,2)*yuv(:,:,2) + t_inv(1,3)*yuv(:,:,3);
rgb(:,:,2) = t_inv(2,1)*yuv(:,:,1) + t_inv(2,2)*yuv(:,:,2) + t_inv(2,3)*yuv(:,:,3);
rgb(:,:,3) = t_inv(3,1)*yuv(:,:,1) + t_inv(3,2)*yuv(:,:,2) + t_inv(3,3)*yuv(:,:,3);
end