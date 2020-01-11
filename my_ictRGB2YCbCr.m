function yuv = my_ictRGB2YCbCr(rgb,t_enc,t_off)
yuv(:,:,1) = t_enc(1,1)*rgb(:,:,1) + t_enc(1,2)*rgb(:,:,2) + t_enc(1,3)*rgb(:,:,3) + t_off(1,:);
yuv(:,:,2) = t_enc(2,1)*rgb(:,:,1) + t_enc(2,2)*rgb(:,:,2) + t_enc(2,3)*rgb(:,:,3) + t_off(2,:);
yuv(:,:,3) = t_enc(3,1)*rgb(:,:,1) + t_enc(3,2)*rgb(:,:,2) + t_enc(3,3)*rgb(:,:,3) + t_off(3,:);
end