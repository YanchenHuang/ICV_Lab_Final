function x = func_Mywaverec2(c,s,varargin)
%WAVEREC2  Multilevel 2-D wavelet reconstruction.
%   WAVEREC2 performs a multilevel 2-D wavelet reconstruction
%   using either a specific wavelet ('wname', see WFILTERS) or
%   specific reconstruction filters (Lo_R and Hi_R).
%
%   X = WAVEREC2(C,S,'wname') reconstructs the matrix X
%   based on the multi-level wavelet decomposition structure
%   [C,S] (see WAVEDEC2).
%
%   For X = WAVEREC2(C,S,Lo_R,Hi_R),
%   Lo_R is the reconstruction low-pass filter and
%   Hi_R is the reconstruction high-pass filter.
%
%   See also APPCOEF2, IDWT2, WAVEDEC2.



% Check arguments.
if errargn(mfilename,nargin,[3:4],nargout,[0:1]), error('*'), end

x = func_Myappcoef2(c,s,varargin{:},0);