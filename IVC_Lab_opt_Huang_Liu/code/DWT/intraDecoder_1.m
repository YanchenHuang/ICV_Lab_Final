function [img_dec]=intraDecoder_1(img, S, Lo_R, Hi_R, level,qScalar)
%input: hoffman decoded coeffs

%output: decoded coeffs
    img_dec = func_SPIHT_Dec(img);
    img_spiht = func_InvDWT(img_dec, S, Lo_R, Hi_R, level);
    img_dec=img_spiht;
end