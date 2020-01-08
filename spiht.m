1�����ȸ��������������
function [T,SnList,RnList,ini_LSP,ini_LIP,ini_LIS,ini_LisFlag]=spihtcoding(DecIm,imDim,codeDim)
% ���� SPIHTCODING() ��SPIHT�㷨�ı���������
% ���������DecIm ����С���ֽ�ϵ������
% imDim ����С���ֽ������
% codeDim �������뼶����
% ���������T ���� ��ʼ��ֵ��T=2^N��N=floor(log2(max{|c(i,j)|}))��c(i,j)ΪС��ϵ�������Ԫ��
% SnList ���� ����ɨ�����λ��
% RnList ���� ��ϸɨ�����λ��
% ini_L* ���� ��ʼϵ�������ϣ���
% LSP����Ҫϵ����
% LIP������Ҫϵ����
% LIS������Ҫ�Ӽ������еı�����D�ͻ�L�ͱ����������
% LisFlag��LIS�и���������ͣ�����D�ͺ�L������
global Mat rMat cMat
% Mat�������С���ֽ�ϵ��������Ϊȫ�ֱ������ڱ������س�����ʹ��
% rMat��cMat��Mat���С���������Ϊȫ�ֱ������ڱ��롢�������س�����ʹ��
%������������������%
% ���C Threshold ���C %
%������������������%
Mat=DecIm;
MaxMat=max(max(abs(Mat)));
N=floor(log2(MaxMat));
T=2^N;
% ��ʽ��N=floor(log2(max{|c(i,j)|}))��c(i,j)ΪС��ϵ�������Ԫ��
%�������������������������C%
% ���C Output Intialization ���C %
%�������������������������C%
SnList=[];
RnList=[];
ini_LSP=[];
ini_LIP=coef_H(imDim);
rlip=size(ini_LIP,1);
ini_LIS=ini_LIP(rlip/4+1:end,:);
rlis=size(ini_LIS,1);
ini_LisFlag(1:rlis)=��D��;
% ini_LSP��ɨ�迪ʼǰ����Ҫϵ������LSP=[]��
% ini_LIP���������������꼯������N��С���ֽ⣬LIP��LL_N,LH_N,HL_N,HH_N ����
% ϵ�������꼯�ϣ�
% ���� COEF_H() ���ڼ����������꼯 H
% ini_LIS����ʼʱ��LIS��LH_N,HL_N,HH_N ����ϵ�������꼯�ϣ���SPIHT�㷨�У�
% LL_N û�к��ӡ�
% ini_LisFlag����ʼʱ��LIS�б�ı����ΪD�͡�
%��������������������������������%
% ���C Coding Input Initialization ���� %
%��������������������������������%
LSP=ini_LSP;
LIP=ini_LIP;
LIS=ini_LIS;
LisFlag=ini_LisFlag;
% ��������ĸ����б������Ӧ�ı��빤���б�
%���������������������C%
% ���C Coding Loop ���� %
%���������������������C%
for d=1:codeDim
%����������������������������%
% ���C Coding Initialization ����- %
%����������������������������%
Sn=[];
LSP_Old=LSP;
% ÿ�����������Sn���Ƕ����ģ���Sn��ʼ��Ϊ�ձ�
% �б�LSP_Old���ڴ洢�ϼ������������Ҫϵ���б�LSP����Ϊ������ϸɨ�������
%��������������������-%
% ���C Sorting Pass ���C %
%��������������������-%
% ���C LIP Scan �����C %
%������������������-%
[Sn,LSP,LIP]=lip_scan(Sn,N,LSP,LIP);
% ���LIP���С��ϵ���������б�LIP��LSP������λ�� Sn
%����������������-%
% ���C LIS Scan ���C %
%����������������-%
[LSP,LIP,LIS,LisFlag,Sn,N]=lis_scan(N,Sn,LSP,LIP,LIS,LisFlag);
% �����Ϊ�����N����Ϊ�����N��1���� out_N=in_N-1
% ���������������Ϊ��һ���뼶������
%������������������������-%
% ���C Refinement Pass ���C %
%������������������������-%
Rn=refinement(N+1,LSP_Old);
% ��ϸɨ�����ڵ�ǰ��ֵT=2^N�£�ɨ����һ���뼶������LSP��������Ϊ(N+1,LSP_Old)��
% ���Ϊ��ϸλ�� Rn
%�����������������������C%
% ���C Output Dataflow ���C %
%�����������������������C%
SnList=[SnList,Sn,7];
RnList=[RnList,Rn,7];
% ���֡�7����Ϊ���ַ�,���ֲ�ͬ���뼶��Rn��Snλ��
end
�����������е��õ����ӳ����У�
COEF_H() �����ڼ����������꼯 H �����ɳ�ʼ��LIP���У�
LIP_SCAN() �����LIP��ĸ��������Ƿ���Ҫ�������б�LIP��LSP������λ�� Sn ��
LIS_SCAN() �����LIS��ĸ��������Ƿ���Ҫ�������б�LIP��LSP��LIS��LisFlag������λ�� Sn ��
REFINEMENT() ����ϸɨ�������������ϸλ�� Rn ��
��1�������Ǽ����������꼯 H �ĳ���
function lp=coef_H(imDim)
% ���� COEF_H() ���ݾ����������rMat��cMat��С���ֽ����imDim�������������꼯 H
% ���������imDim ���� С���ֽ����,Ҳ�ɼ��� N
% ���������lp ���� rMat*cMat����N��ֽ��LL_N,LH_N,HL_N,HH_N ����ϵ�������꼯��
global rMat cMat
% rMat��cMat��Mat���С���������Ϊȫ�ֱ������ڱ��롢�������س�����ʹ��
row=rMat/2^(imDim-1);
col=cMat/2^(imDim-1);
% row��col�� LL_N,LH_N,HL_N,HH_N ��ɵľ�����С�����
lp=listorder(row,col,1,1);
% ��Ϊ LIP �� LIS ��Ԫ��(r,c)������˳���� EZW �����ṹ��ɨ��˳����ͬ
% ֱ�ӵ��ú��� LISTORDER() ���ɻ�ð�EZWɨ��˳�����е�LIP�б�
��2����������˺��� LISTORDER() ����ȡ��EZWɨ��˳�����е�LIP�б������Ǹú����ĳ�����룺
function lsorder=listorder(mr,mc,pr,pc)
% ���� LISTORDER() ���ɰ���Z���͵ݹ�ṹ���е������б�
% �����ݹ�ԭ����һ��mr*mc�ľ��������Ͻ�Ԫ�ص�����Ϊ(pr,pc)�����Ƚ����󰴡��
% ���ͷֳ��ĸ��Եȵ��Ӿ���ÿ���Ӿ�����С�������Ϊmr/2��mc/2�����Ͻ�Ԫ�ص�����
% ���ϵ��¡������ҷֱ�Ϊ(pr,pc)��(pr,pc+mc/2)��(pr+mr/2,pc)��(pr+mr/2,pc+mc/2)��
% ��ÿ���Ӿ����ٷ��ѳ��ĸ�������˵ݹ������ȥ��ֱ����С���������������2����ȡ��С
% ������ĸ��������ֵ��Ȼ�������ϻ��ݣ����ɵõ�����Z���͵ݹ�ṹ���е������б�
lso=[pr,pc;pr,pc+mc/2;pr+mr/2,pc;pr+mr/2,pc+mc/2];
% �б�lso�Ǹ�������ѳ��ĸ��Ӿ���󣬸��Ӿ������Ͻ�Ԫ������ļ���
mr=mr/2;
mc=mc/2;
% �Ӿ�����������Ǹ������һ��
lm1=[];lm2=[];lm3=[];lm4=[];
if (mr>1)&&(mc>1)
% ����Z���ͽṹ�ݹ�
ls1=listorder(mr,mc,lso(1,1),lso(1,2));
lm1=[lm1;ls1];
ls2=listorder(mr,mc,lso(2,1),lso(2,2));
lm2=[lm2;ls2];
ls3=listorder(mr,mc,lso(3,1),lso(3,2));
lm3=[lm3;ls3];
ls4=listorder(mr,mc,lso(4,1),lso(4,2));
lm4=[lm4;ls4];
end
lsorder=[lso;lm1;lm2;lm3;lm4];
% �ĸ��Ӿ�������ݹ���ݵ�������ʱ���б�lsorder��ͷ�ĸ�����ֵΪ�б�lso��Ԫ��
% ���ĸ�����ֵ�����ĸ����Ӿ��������Ԫ�����ص���������ȥ
% ������������б���length(lsorder)������Ԫ�ظ���mr*mc*4�����ʱ��
% ��˵���������ص�������
len=length(lsorder);
lsorder=lsorder(len-mr*mc*4+1:len,:);
���ĸ���SPIHT���������ɨ����룬����ɨ���ΪLIP����ɨ���LIS����ɨ���������裬����LIS����ɨ���Ϊ���ӣ��ڱ��ʱ���׳��ִ���Ҫ����ע�⡣
2��LIP����ɨ�����
function [Sn,LSP,LIP]=lip_scan(Sn,N,LSP,LIP)
% ���� LIP_SCAN() ���LIP��ĸ��������Ƿ���Ҫ�������б�LIP��LSP������λ�� Sn
% ���������Sn ���� ������������λ����Ϊ�ձ�
% N ���� ����������ֵ��ָ��
% LSP ���� ��һ���������ɵ���Ҫϵ���б�
% LIP ���� ��һ���������ɵĲ���Ҫϵ���б�
% ���������Sn ���� ����һ���������ɵ�LIP�б�ɨ�����µ�����λ��
% LSP ���� ����һ���������ɵ�LIP�б�ɨ�����µ���Ҫϵ���б�
% LIP ���� ������LIPɨ�账�����µĲ���Ҫϵ���б�
global Mat
% Mat�������С���ֽ�ϵ��������Ϊȫ�ֱ������ڱ������س�����ʹ��
rlip=size(LIP,1);
% r ��ָ�� LIP ��ǰ�������λ�õ�ָ��
r=1;
% ����ѭ���������б� LIP �ı���仯�����ʺ��� for ѭ�����ʲ��� while ѭ��
while r<=rlip
% ���뵱ǰ���������ֵ
rN=LIP(r,1);
cN=LIP(r,2);
% ���� SNOUT() �������жϸñ����Ƿ���Ҫ
if SnOut(LIP(r,:),N)
% ����Ҫ�������롮1���� Sn
Sn=[Sn,1];
% �����������š�1����0���� Sn
if Mat(rN,cN)>=0
Sn=[Sn,1];
else
Sn=[Sn,0];
end
% ���ñ�����ӵ���Ҫϵ���б� LSP
LSP=[LSP;LIP(r,:)];
% ���ñ���� LIP ��ɾ��
LIP(r,:)=[];
else
% ������Ҫ�������롮0���� Sn
Sn=[Sn,0];
% ��ָ��ָ����һ������
r=r+1;
end
% �жϵ�ǰ LIP �ı�
rlip=size(LIP,1);
end
3��LIS����ɨ�����
function [LSP,LIP,LIS,LisFlag,Sn,N]=lis_scan(N,Sn,LSP,LIP,LIS,LisFlag)
% ���� LIS_SCAN() ���LIS��ĸ��������Ƿ���Ҫ�������б�LIP��LSP��LIS��LisFlag������λ�� Sn
% ���������N ���� ����������ֵ��ָ��
% Sn ���� ������������λ����Ϊ�ձ�
% LSP ���� ��һ���������ɵ���Ҫϵ���б�
% LIP ���� ��һ���������ɵĲ���Ҫϵ���б�
% LIS ���� ��һ���������ɵĲ���Ҫ�Ӽ��б�
% LisFlag ���� ��һ���������ɵĲ���Ҫ�Ӽ����������б�
% ���������LSP ���� ����LIS�б�ɨ�����µ���Ҫϵ���б�
% LIP ���� ������LISɨ�账�����µĲ���Ҫϵ���б�
% LIS ���� ����LIS�б�ɨ�����µĲ���Ҫ�Ӽ��б�
% LisFlag ���� ����LIS�б�ɨ�����µĲ���Ҫ�Ӽ����������б�
% Sn ���� ����LIS�б�ɨ�����µ�����λ��
% N ���� ��һ��������ֵ��ָ��
global Mat rMat cMat
% Mat�������С���ֽ�ϵ��������Ϊȫ�ֱ������ڱ������س�����ʹ��
% rMat��cMat��Mat���С���������Ϊȫ�ֱ������ڱ��롢�������س�����ʹ��
% ���뵱ǰ LIS �ı�
rlis=size(LIS,1);
% ls ��ָ�� LIS ��ǰ����λ�õ�ָ�룬��ʼλ��Ϊ1
ls=1;
% ����ѭ���������б� LIS �ı���仯�����ʺ��� for ѭ�����ʲ��� while ѭ��
while ls<=rlis
% ���뵱ǰ LIS ���������
switch LisFlag(ls)
% ��D�������������Ӻͷ�ֱϵ����
case ��D��
% ����ñ��������ֵ
rP=LIS(ls,1);
cP=LIS(ls,2);
% ���ɡ�D����������
chD=coef_DOL(rP,cP,��D��);
% �жϸ��������Ƿ���Ҫ
isImt=SnOut(chD,N);
if isImt
% �������������Ҫ�������롮1���� Sn
Sn=[Sn,1];
% ���ɸñ���ĺ�����
chO=coef_DOL(rP,cP,��O��);
% �ֱ��ж�ÿ�����ӵ���Ҫ��
for r=1:4
% �жϸú��ӵ���Ҫ�Ժ���������
[isImt,Sign]=SnOut(chO(r,:),N);
if isImt
% �����Ҫ�������롮1����������־�� Sn
Sn=[Sn,1];
if Sign
Sn=[Sn,1];
else
Sn=[Sn,0];
end
% ���ú�����ӵ���Ҫϵ���б� LSP
LSP=[LSP;chO(r,:)];
else
% �������Ҫ�������롮0���� Sn
Sn=[Sn,0];
% ���ú�����ӵ�����Ҫϵ���б� LIP
% ������ֵ�²���Ҫ��ϵ������һ�������п�������Ҫ��
LIP=[LIP;chO(r,:)];
end
end
% ���ɸñ���ķ�ֱϵ������
chL=coef_DOL(rP,cP,��L��);
if ~isempty(chL)
% �����L�������ǿգ��򽫸ñ�����ӵ��б� LIS ��β�˵ȴ�ɨ��
LIS=[LIS;LIS(ls,:)];
% �������͸���Ϊ��L����
LisFlag=[LisFlag,��L��];
% ���ˣ��ñ���ġ�D����LISɨ���������LIS��ɾ����������ͷ�
LIS(ls,:)=[];
LisFlag(ls)=[];
else
% �����L������Ϊ�ռ�
% ��ñ���ġ�D����LISɨ���������LIS��ɾ����������ͷ�
LIS(ls,:)=[];
LisFlag(ls)=[];
end
else
% ����ñ���ġ�D��������������Ҫ�������롮0���� Sn
Sn=[Sn,0];
% ��ָ��ָ����һ�� LIS ����
ls=ls+1;
end
% ���µ�ǰ LIS �ı���ת����һ�����ɨ��
rlis=size(LIS,1);
case ��L��
% �ԡ�L�����������жϺ��ӵ���Ҫ��
% ����ñ��������ֵ
rP=LIS(ls,1);
cP=LIS(ls,2);
% ���ɡ�L����������
chL=coef_DOL(rP,cP,��L��);
% �жϸ��������Ƿ���Ҫ
isImt=SnOut(chL,N);
if isImt
% �������������Ҫ�������롮1���� Sn
Sn=[Sn,1];
% ���ɸñ���ĺ�����
chO=coef_DOL(rP,cP,��O��);
% ���á�L�������� LIS ��ɾ��
LIS(ls,:)=[];
LisFlag(ls)=[];
% ��������ĸ�������ӵ� LIS �Ľ�β�����Ϊ��D�������
LIS=[LIS;chO(1:4,:)];
LisFlag(end+1:end+4)=��D��;
else
% ������������ǲ���Ҫ�ģ������롮0���� Sn
Sn=[Sn,0];
% ��ָ��ָ����һ�� LIS ����
ls=ls+1;
end
% ���µ�ǰ LIS �ı���ת����һ�����ɨ��
rlis=size(LIS,1);
end
end
% �� LIS ��ɨ���������������ֵ��ָ����1��׼��������һ������
N=N-1;
LIS����ɨ������е��õ��ӳ����У�
COEF_DOL() �����������������͡�type���������(r,c)�������б�
SNOUT() �����ݱ�����ֵָ�� N �ж����꼯 coefSet �Ƿ���Ҫ��
��1�����������ɳ���
function chList=coef_DOL(r,c,type)
% ���� COEF_DOL() ���������������͡�type���������(r,c)�������б�
% ���������r,c ���� С��ϵ��������ֵ
% type ���� ������������
% ��D������(r,c)����������������ӣ�
% ��O������(r,c)�����к���
% ��L������(r,c)�����з�ֱϵ���L(r,c)=D(r,c)-O(r,c)
% ���������chList ���� ��(r,c)�ġ�type���������б�
global Mat rMat cMat
% Mat�������С���ֽ�ϵ��������Ϊȫ�ֱ������ڱ������س�����ʹ��
% rMat��cMat��Mat���С���������Ϊȫ�ֱ������ڱ��롢�������س�����ʹ��
[Dch,Och]=childMat(r,c);
% ���� CHILDMAT() ���ص�(r,c)�ġ�D���ͺ͡�O���������б�
Lch=setdiff(Dch,Och,��rows��);
% ���ݹ�ϵʽ L(r,c)=D(r,c)-O(r,c)�����L���������б�
% Matlab���� SETDIFF(A,B)���������ͬ��������������A��B�У�A�ж�B�޵�Ԫ�ؼ���
% �������������type��ѡ���������
switch type
case ��D��
chList=Dch;
case ��O��
chList=Och;
case ��L��
chList=Lch;
end
function [trAll,trChl]=childMat(trRows,trCols)
% ���� CHILDMAT() �������������ֵtrRows��trCols �����ȫ������ trAll��
% ���а��������� trChl�����⣬�����㷨ԭ����Ҫ�ж��������Ƿ�ȫΪ�㣬
% ��Ϊȫ�㣬��trAll��trChl��Ϊ�ձ�
global Mat rMat cMat
% Mat�������С���ֽ�ϵ��������Ϊȫ�ֱ������ڱ������س�����ʹ��
% rMat��cMat��Mat���С���������Ϊȫ�ֱ������ڱ��롢�������س�����ʹ��
trAll=treeMat(trRows,trCols);
% ���ú��� treeMat() ���ɸõ���������������
trZero=1;
% �ñ��� trZero ����Ǹõ��Ƿ���з�������
rA=size(trAll,1);
% ��������� trAll ����ϵ��ֵ��Ϊ�㣬�� trZero=0����ʾ�õ���з�������
for r=1:rA
if Mat(trAll(r,1),trAll(r,2))~=0
trZero=0;
break;
end
end
if trZero
trAll=[];
trChl=[];
else
trChl=trAll(1:4,:);
% ���� treeMat() �����ȫ�������� trAll ͷ��λԪ�ؾ������Ӧ�ĺ�����
end
������õĺ���treeMat() ��EZW�㷨��ʹ�õĳ�����һ���ģ�����Ͳ�д�����ˣ���ϸ������μ���Ƕ��ʽС��������EZW���㷨�Ĺ�������Matlab���루1������ɨ��������
��2��ϵ������Ҫ���б����
function [isImt,Sign]=SnOut(coefSet,N)
% ���� SNOUT() ���ݱ�����ֵָ�� N �ж����꼯 coefSet �Ƿ���Ҫ isImt ���Ե�Ԫ��
% ��ϵ���������Ԫ�ص��������� Sign ��
global Mat
% Mat�������С���ֽ�ϵ��������Ϊȫ�ֱ������ڱ������س�����ʹ��
allMat=[];
isImt=0;
Sign=0;
% Ĭ�����꼯�ǲ���Ҫ�ģ�����λԪ���Ǹ�ֵ
rSet=size(coefSet,1);
% ��ȡ���꼯�и�Ԫ�ص�ϵ��ֵ
for r=1:rSet
allMat(r)=Mat(coefSet(r,1),coefSet(r,2));
if abs(allMat(r))>=2^N
isImt=1;
break;
end
end
% �Ե���Ԫ�ص����꼯���жϸ�Ԫ�ص���������
% ���ں��� childMat() ������ȫ��ĵ�᷵�ؿձ�����Ҫ���allMat�Ƿ�Ϊ��
if ~isempty(allMat)&&(allMat(1)>=0)
Sign=1;
end
���ĸ���SPIHT����ľ�ϸɨ��������а���һ���ܹ�����С����ʮ������ת��Ϊ�����Ʊ�ʾ�ĺ��������ת����������ʵ�����⾫�ȵĶ�����ת�����ر��ǽ�С������ת��Ϊ�����Ʊ�ʾ��ϣ��������Ҫ������������������һƪ���½�����SPIHT�Ľ���������ע�������£���ӭ Email ��ϵ������
4����ϸɨ�����
function Rn=refinement(N,LSP_Old)
% ���� REFINEMENT()Ϊ��ϸ������򣬶���һ�������������Ҫϵ���б�LSP_Old����ȡÿ��
% ������ӦС��ϵ������ֵ�Ķ����Ʊ�ʾ��������е�N����Ҫ��λ������Ӧ�� 2^N ��������
% ���������N ���� ����������ֵ��ָ��
% LSP_Old ���� ��һ�������������Ҫϵ���б�
% ���������Rn ���� ��ϸɨ�����λ��
global Mat
% Mat�������С���ֽ�ϵ��������Ϊȫ�ֱ������ڱ������س�����ʹ��
Rn=[];
% ÿ����ϸɨ�迪ʼʱ��Rn ��Ϊ�ձ�
% LSP_Old �ǿ�ʱ��ִ�о�ϸɨ�����
if ~isempty(LSP_Old)
rlsp=size(LSP_Old,1);
% ��ȡ LSP_Old �ı����������ÿ���������ɨ��
for r=1:rlsp
tMat=Mat(LSP_Old(r,1),LSP_Old(r,2));
% ��ȡ�ñ����Ӧ��С��ϵ��ֵ
[biLSP,Np]=fracnum2bin(abs(tMat),N);
% ���� FRACNUM2BIN() ���ݾ�ϸɨ���Ӧ��Ȩλ N ���������ʮ��������ת��Ϊ����������
% �������Ϊ�����Ʊ�ʾ�б� biLSP �� ȨλN�����Ȩλ�ľ��� Np ��
Rn=[Rn,biLSP(Np)];
% biLSP(Np)��ΪС��ϵ������ֵ�Ķ����Ʊ�ʾ�е�N����Ҫ��λ
end
end
��1��ʮ������ת��Ϊ�����Ʊ�ʾ�ĳ���
function [binlist,qLpoint]=fracnum2bin(num,qLevel)
% ���� FRACNUM2BIN() ���ݾ�ϸɨ���Ӧ��Ȩλ N ���������ʮ��������ת��Ϊ����������
% ������������λС����ʮ��������Matlab�еĺ��� dec2bin()��dec2binvec()ֻ�ܽ�ʮ
% ����������������ת��Ϊ�����Ʊ�ʾ����С��������ת����
%
% ���������num ���� �Ǹ���ʮ������
% qLevel ���� ����ת�����ȣ�Ҳ�����Ǿ�ϸɨ���Ӧ��Ȩλ N
% ���������biLSP ���� �����Ʊ�ʾ�б�
% Np ���� ȨλN�����Ȩλ�ľ��룬N Ҳ�Ǳ���������ֵ��ָ��
intBin=dec2binvec(num);
% ������Matlab����dec2binvec()��ȡ�������ֵĶ����Ʊ�ʾintBin����λ��ǰ����λ�ں�
intBin=intBin(end:-1:1);
% ���ݸ���ϰ�ߣ��������Ʊ�ʾת��Ϊ��λ��ǰ����λ�ں�
lenIB=length(intBin);
% ��������Ʊ�ʾ�ĳ���
decpart=num-floor(num);
% ���С������
decBin=[];
% С�����ֵĶ����Ʊ�ʾ��ʼ��Ϊ�ձ�
% ������������Ҫ������ܵĶ����Ʊ�ʾ�б�
if (qLevel+1)>lenIB
% ����������ȸ����������ֵĶ������볤�������Ϊ��ֵ�б�
binlist=zeros(1,qLevel+1);
qLpoint=1;
elseif qLevel>=0
% �����������������Ȩλ��������������ֵĶ����Ʊ�ʾintBin
% ����ת��С�����֣�ͬʱ����������������Ȩλ�ľ���Np
binlist=intBin;
binlist(lenIB-qLevel+1:end)=0;
qLpoint=lenIB-qLevel;
elseif qLevel<0
% �������������С��Ȩλ������ת��С������
N=-1;
while N>=qLevel
% С�����ֵ�ת��ֻ����е��������ȴ�
res=decpart-2^N;
if res==0
decBin=[decBin,1];
decBin(end+1:-qLevel)=0;
% ���С�����ֵ�ת�����ʱ��δ�ﵽ�����������ڵ�Ȩλ������
break;
elseif res>0
decBin=[decBin,1];
decpart=res;
N=N-1;
else
decBin=[decBin,0];
N=N-1;
end
end
binlist=[intBin,decBin];
qLpoint=lenIB-qLevel;
% ����������ֺ�С�����ֵĶ����Ʊ�ʾintBin,decBin���Լ��������������Ȩλ�ľ���Np
end