1、首先给出编码的主程序
function [T,SnList,RnList,ini_LSP,ini_LIP,ini_LIS,ini_LisFlag]=spihtcoding(DecIm,imDim,codeDim)
% 函数 SPIHTCODING() 是SPIHT算法的编码主程序
% 输入参数：DecIm ——小波分解系数矩阵；
% imDim ——小波分解层数；
% codeDim ——编码级数。
% 输出参数：T —— 初始阈值，T=2^N，N=floor(log2(max{|c(i,j)|}))，c(i,j)为小波系数矩阵的元素
% SnList —— 排序扫描输出位流
% RnList —— 精细扫描输出位流
% ini_L* —— 初始系数（集合）表
% LSP：重要系数表
% LIP：不重要系数表
% LIS：不重要子集表，其中的表项是D型或L型表项的树根点
% LisFlag：LIS中各表项的类型，包括D型和L型两种
global Mat rMat cMat
% Mat是输入的小波分解系数矩阵，作为全局变量，在编码的相关程序中使用
% rMat、cMat是Mat的行、列数，作为全局变量，在编码、解码的相关程序中使用
%—————————%
% —– Threshold —– %
%—————————%
Mat=DecIm;
MaxMat=max(max(abs(Mat)));
N=floor(log2(MaxMat));
T=2^N;
% 公式：N=floor(log2(max{|c(i,j)|}))，c(i,j)为小波系数矩阵的元素
%————————————–%
% —– Output Intialization —– %
%————————————–%
SnList=[];
RnList=[];
ini_LSP=[];
ini_LIP=coef_H(imDim);
rlip=size(ini_LIP,1);
ini_LIS=ini_LIP(rlip/4+1:end,:);
rlis=size(ini_LIS,1);
ini_LisFlag(1:rlis)=’D’;
% ini_LSP：扫描开始前无重要系数，故LSP=[]；
% ini_LIP：所有树根的坐标集，对于N层小波分解，LIP是LL_N,LH_N,HL_N,HH_N 所有
% 系数的坐标集合；
% 函数 COEF_H() 用于计算树根坐标集 H
% ini_LIS：初始时，LIS是LH_N,HL_N,HH_N 所有系数的坐标集合；在SPIHT算法中，
% LL_N 没有孩子。
% ini_LisFlag：初始时，LIS列表的表项均为D型。
%————————————————%
% —– Coding Input Initialization —— %
%————————————————%
LSP=ini_LSP;
LIP=ini_LIP;
LIS=ini_LIS;
LisFlag=ini_LisFlag;
% 将待输出的各项列表存入相应的编码工作列表
%——————————–%
% —– Coding Loop —— %
%——————————–%
for d=1:codeDim
%——————————————%
% —– Coding Initialization ——- %
%——————————————%
Sn=[];
LSP_Old=LSP;
% 每级编码产生的Sn都是独立的，故Sn初始化为空表
% 列表LSP_Old用于存储上级编码产生的重要系数列表LSP，作为本级精细扫描的输入
%——————————-%
% —– Sorting Pass —– %
%——————————-%
% —– LIP Scan ——– %
%—————————-%
[Sn,LSP,LIP]=lip_scan(Sn,N,LSP,LIP);
% 检查LIP表的小波系数，更新列表LIP、LSP和排序位流 Sn
%————————-%
% —– LIS Scan —– %
%————————-%
[LSP,LIP,LIS,LisFlag,Sn,N]=lis_scan(N,Sn,LSP,LIP,LIS,LisFlag);
% 这里，作为输出的N比作为输入的N少1，即 out_N=in_N-1
% 各项输出参数均作为下一编码级的输入
%————————————-%
% —– Refinement Pass —– %
%————————————-%
Rn=refinement(N+1,LSP_Old);
% 精细扫描是在当前阈值T=2^N下，扫描上一编码级产生的LSP，故输入为(N+1,LSP_Old)，
% 输出为精细位流 Rn
%———————————–%
% —– Output Dataflow —– %
%———————————–%
SnList=[SnList,Sn,7];
RnList=[RnList,Rn,7];
% 数字‘7’作为区分符,区分不同编码级的Rn、Sn位流
end
编码主程序中调用到的子程序有：
COEF_H() ：用于计算树根坐标集 H ，生成初始的LIP队列；
LIP_SCAN() ：检查LIP表的各个表项是否重要，更新列表LIP、LSP和排序位流 Sn ；
LIS_SCAN() ：检查LIS表的各个表项是否重要，更新列表LIP、LSP、LIS、LisFlag和排序位流 Sn ；
REFINEMENT() ：精细扫描编码程序，输出精细位流 Rn 。
（1）下面是计算树根坐标集 H 的程序
function lp=coef_H(imDim)
% 函数 COEF_H() 根据矩阵的行列数rMat、cMat和小波分解层数imDim来计算树根坐标集 H
% 输入参数：imDim —— 小波分解层数,也可记作 N
% 输出参数：lp —— rMat*cMat矩阵经N层分解后，LL_N,LH_N,HL_N,HH_N 所有系数的坐标集合
global rMat cMat
% rMat、cMat是Mat的行、列数，作为全局变量，在编码、解码的相关程序中使用
row=rMat/2^(imDim-1);
col=cMat/2^(imDim-1);
% row、col是 LL_N,LH_N,HL_N,HH_N 组成的矩阵的行、列数
lp=listorder(row,col,1,1);
% 因为 LIP 和 LIS 中元素(r,c)的排列顺序与 EZW 零树结构的扫描顺序相同
% 直接调用函数 LISTORDER() 即可获得按EZW扫描顺序排列的LIP列表
（2）这里调用了函数 LISTORDER() 来获取按EZW扫描顺序排列的LIP列表，以下是该函数的程序代码：
function lsorder=listorder(mr,mc,pr,pc)
% 函数 LISTORDER() 生成按‘Z’型递归结构排列的坐标列表
% 函数递归原理：对一个mr*mc的矩阵，其左上角元素的坐标为(pr,pc)；首先将矩阵按“田”
% 字型分成四个对等的子矩阵，每个子矩阵的行、列数均为mr/2、mc/2，左上角元素的坐标
% 从上到下、从左到右分别为(pr,pc)、(pr,pc+mc/2)、(pr+mr/2,pc)、(pr+mr/2,pc+mc/2)。
% 把每个子矩阵再分裂成四个矩阵，如此递归分裂下去，直至最小矩阵的行列数等于2，获取最小
% 矩阵的四个点的坐标值，然后逐步向上回溯，即可得到按‘Z’型递归结构排列的坐标列表。
lso=[pr,pc;pr,pc+mc/2;pr+mr/2,pc;pr+mr/2,pc+mc/2];
% 列表lso是父矩阵分裂成四个子矩阵后，各子矩阵左上角元素坐标的集合
mr=mr/2;
mc=mc/2;
% 子矩阵的行列数是父矩阵的一半
lm1=[];lm2=[];lm3=[];lm4=[];
if (mr>1)&&(mc>1)
% 按‘Z’型结构递归
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
% 四个子矩阵结束递归回溯到父矩阵时，列表lsorder的头四个坐标值为列表lso的元素
% 这四个坐标值与后面的各个子矩阵的坐标元素有重叠，故需消去
% 当函数输出的列表长度length(lsorder)与矩阵的元素个数mr*mc*4不相等时，
% 就说明有坐标重叠发生。
len=length(lsorder);
lsorder=lsorder(len-mr*mc*4+1:len,:);
本文给出SPIHT编码的排序扫描代码，排序扫描分为LIP队列扫描和LIS队列扫描两个步骤，其中LIS队列扫描较为复杂，在编程时容易出现错误，要倍加注意。
2、LIP队列扫描程序
function [Sn,LSP,LIP]=lip_scan(Sn,N,LSP,LIP)
% 函数 LIP_SCAN() 检查LIP表的各个表项是否重要，更新列表LIP、LSP和排序位流 Sn
% 输入参数：Sn —— 本级编码排序位流，为空表
% N —— 本级编码阈值的指数
% LSP —— 上一级编码生成的重要系数列表
% LIP —— 上一级编码生成的不重要系数列表
% 输出参数：Sn —— 对上一级编码生成的LIP列表扫描后更新的排序位流
% LSP —— 对上一级编码生成的LIP列表扫描后更新的重要系数列表
% LIP —— 经本级LIP扫描处理后更新的不重要系数列表
global Mat
% Mat是输入的小波分解系数矩阵，作为全局变量，在编码的相关程序中使用
rlip=size(LIP,1);
% r 是指向 LIP 当前读入表项位置的指针
r=1;
% 由于循环过程中列表 LIP 的表长会变化，不适合用 for 循环，故采用 while 循环
while r<=rlip
% 读入当前表项的坐标值
rN=LIP(r,1);
cN=LIP(r,2);
% 调用 SNOUT() 函数来判断该表项是否重要
if SnOut(LIP(r,:),N)
% 若重要，则输入‘1’到 Sn
Sn=[Sn,1];
% 输入正负符号‘1’或‘0’到 Sn
if Mat(rN,cN)>=0
Sn=[Sn,1];
else
Sn=[Sn,0];
end
% 将该表项添加到重要系数列表 LSP
LSP=[LSP;LIP(r,:)];
% 将该表项从 LIP 中删除
LIP(r,:)=[];
else
% 若不重要，则输入‘0’到 Sn
Sn=[Sn,0];
% 将指针指向下一个表项
r=r+1;
end
% 判断当前 LIP 的表长
rlip=size(LIP,1);
end
3、LIS队列扫描程序
function [LSP,LIP,LIS,LisFlag,Sn,N]=lis_scan(N,Sn,LSP,LIP,LIS,LisFlag)
% 函数 LIS_SCAN() 检查LIS表的各个表项是否重要，更新列表LIP、LSP、LIS、LisFlag和排序位流 Sn
% 输入参数：N —— 本级编码阈值的指数
% Sn —— 本级编码排序位流，为空表
% LSP —— 上一级编码生成的重要系数列表
% LIP —— 上一级编码生成的不重要系数列表
% LIS —— 上一级编码生成的不重要子集列表
% LisFlag —— 上一级编码生成的不重要子集表项类型列表
% 输出参数：LSP —— 本级LIS列表扫描后更新的重要系数列表
% LIP —— 经本级LIS扫描处理后更新的不重要系数列表
% LIS —— 本级LIS列表扫描后更新的不重要子集列表
% LisFlag —— 本级LIS列表扫描后更新的不重要子集表项类型列表
% Sn —— 本级LIS列表扫描后更新的排序位流
% N —— 下一级编码阈值的指数
global Mat rMat cMat
% Mat是输入的小波分解系数矩阵，作为全局变量，在编码的相关程序中使用
% rMat、cMat是Mat的行、列数，作为全局变量，在编码、解码的相关程序中使用
% 读入当前 LIS 的表长
rlis=size(LIS,1);
% ls 是指向 LIS 当前表项位置的指针，初始位置为1
ls=1;
% 由于循环过程中列表 LIS 的表长会变化，不适合用 for 循环，故采用 while 循环
while ls<=rlis
% 读入当前 LIS 表项的类型
switch LisFlag(ls)
% ‘D’类表项，包含孩子和非直系子孙
case ‘D’
% 读入该表项的坐标值
rP=LIS(ls,1);
cP=LIS(ls,2);
% 生成‘D’型子孙树
chD=coef_DOL(rP,cP,’D’);
% 判断该子孙树是否重要
isImt=SnOut(chD,N);
if isImt
% 如果该子孙树重要，就输入‘1’到 Sn
Sn=[Sn,1];
% 生成该表项的孩子树
chO=coef_DOL(rP,cP,’O’);
% 分别判断每个孩子的重要性
for r=1:4
% 判断该孩子的重要性和正负符号
[isImt,Sign]=SnOut(chO(r,:),N);
if isImt
% 如果重要，就输入‘1’和正负标志到 Sn
Sn=[Sn,1];
if Sign
Sn=[Sn,1];
else
Sn=[Sn,0];
end
% 将该孩子添加到重要系数列表 LSP
LSP=[LSP;chO(r,:)];
else
% 如果不重要，就输入‘0’到 Sn
Sn=[Sn,0];
% 将该孩子添加到不重要系数列表 LIP
% 本级阈值下不重要的系数在下一级编码中可能是重要的
LIP=[LIP;chO(r,:)];
end
end
% 生成该表项的非直系子孙树
chL=coef_DOL(rP,cP,’L’);
if ~isempty(chL)
% 如果‘L’型树非空，则将该表项添加到列表 LIS 的尾端等待扫描
LIS=[LIS;LIS(ls,:)];
% 表项类型更改为‘L’型
LisFlag=[LisFlag,’L’];
% 至此，该表项的‘D’型LIS扫描结束，在LIS中删除该项及其类型符
LIS(ls,:)=[];
LisFlag(ls)=[];
else
% 如果‘L’型树为空集
% 则该表项的‘D’型LIS扫描结束，在LIS中删除该项及其类型符
LIS(ls,:)=[];
LisFlag(ls)=[];
end
else
% 如果该表项的‘D’型子孙树不重要，就输入‘0’到 Sn
Sn=[Sn,0];
% 将指针指向下一个 LIS 表项
ls=ls+1;
end
% 更新当前 LIS 的表长，转入下一表项的扫描
rlis=size(LIS,1);
case ‘L’
% 对‘L’类表项，不需判断孩子的重要性
% 读入该表项的坐标值
rP=LIS(ls,1);
cP=LIS(ls,2);
% 生成‘L’型子孙树
chL=coef_DOL(rP,cP,’L’);
% 判断该子孙树是否重要
isImt=SnOut(chL,N);
if isImt
% 如果该子孙树重要，就输入‘1’到 Sn
Sn=[Sn,1];
% 生成该表项的孩子树
chO=coef_DOL(rP,cP,’O’);
% 将该‘L’类表项从 LIS 中删除
LIS(ls,:)=[];
LisFlag(ls)=[];
% 将表项的四个孩子添加到 LIS 的结尾，标记为‘D’类表项
LIS=[LIS;chO(1:4,:)];
LisFlag(end+1:end+4)=’D’;
else
% 如果该子孙树是不重要的，就输入‘0’到 Sn
Sn=[Sn,0];
% 将指针指向下一个 LIS 表项
ls=ls+1;
end
% 更新当前 LIS 的表长，转入下一表项的扫描
rlis=size(LIS,1);
end
end
% 对 LIS 的扫描结束，将本级阈值的指数减1，准备进入下一级编码
N=N-1;
LIS队列扫描程序中调用的子程序有：
COEF_DOL() ：根据子孙树的类型’type’来计算点(r,c)的子孙列表；
SNOUT() ：根据本级阈值指数 N 判断坐标集 coefSet 是否重要；
（1）子孙树生成程序
function chList=coef_DOL(r,c,type)
% 函数 COEF_DOL() 根据子孙树的类型’type’来计算点(r,c)的子孙列表
% 输入参数：r,c —— 小波系数的坐标值
% type —— 子孙树的类型
% ‘D’：点(r,c)的所有子孙（包括孩子）
% ‘O’：点(r,c)的所有孩子
% ‘L’：点(r,c)的所有非直系子孙，L(r,c)=D(r,c)-O(r,c)
% 输出参数：chList —— 点(r,c)的’type’型子孙列表
global Mat rMat cMat
% Mat是输入的小波分解系数矩阵，作为全局变量，在编码的相关程序中使用
% rMat、cMat是Mat的行、列数，作为全局变量，在编码、解码的相关程序中使用
[Dch,Och]=childMat(r,c);
% 函数 CHILDMAT() 返回点(r,c)的’D’型和’O’型子孙列表
Lch=setdiff(Dch,Och,’rows’);
% 根据关系式 L(r,c)=D(r,c)-O(r,c)求出’L’型子孙列表
% Matlab函数 SETDIFF(A,B)计算具有相同列数的两个矩阵A、B中，A有而B无的元素集合
% 根据输入参数’type’选择输出参数
switch type
case ‘D’
chList=Dch;
case ‘O’
chList=Och;
case ‘L’
chList=Lch;
end
function [trAll,trChl]=childMat(trRows,trCols)
% 函数 CHILDMAT() 根据输入的坐标值trRows、trCols 输出其全体子孙 trAll，
% 其中包括孩子树 trChl；另外，根据算法原理，还要判断子孙树是否全为零，
% 若为全零，则trAll、trChl均为空表
global Mat rMat cMat
% Mat是输入的小波分解系数矩阵，作为全局变量，在编码的相关程序中使用
% rMat、cMat是Mat的行、列数，作为全局变量，在编码、解码的相关程序中使用
trAll=treeMat(trRows,trCols);
% 调用函数 treeMat() 生成该点的子孙树坐标队列
trZero=1;
% 用变量 trZero 来标记该点是否具有非零子孙
rA=size(trAll,1);
% 如果子孙树 trAll 中有系数值不为零，则 trZero=0，表示该点具有非零子孙
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
% 函数 treeMat() 输出的全体子孙树 trAll 头四位元素就组成相应的孩子树
end
上面调用的函数treeMat() 与EZW算法中使用的程序是一样的，这里就不写出来了，详细代码请参见《嵌入式小波零树（EZW）算法的过程详解和Matlab代码（1）构建扫描次序表》。
（2）系数集重要性判别程序
function [isImt,Sign]=SnOut(coefSet,N)
% 函数 SNOUT() 根据本级阈值指数 N 判断坐标集 coefSet 是否重要 isImt ，对单元素
% 的系数集输出该元素的正负符号 Sign 。
global Mat
% Mat是输入的小波分解系数矩阵，作为全局变量，在编码的相关程序中使用
allMat=[];
isImt=0;
Sign=0;
% 默认坐标集是不重要的，且首位元素是负值
rSet=size(coefSet,1);
% 读取坐标集中各元素的系数值
for r=1:rSet
allMat(r)=Mat(coefSet(r,1),coefSet(r,2));
if abs(allMat(r))>=2^N
isImt=1;
break;
end
end
% 对单个元素的坐标集，判断该元素的正负符号
% 由于函数 childMat() 对子孙全零的点会返回空表，所以要检查allMat是否为空
if ~isempty(allMat)&&(allMat(1)>=0)
Sign=1;
end
本文给出SPIHT编码的精细扫描程序，其中包括一个能够将带小数的十进制数转换为二进制表示的函数，这个转换函数可以实现任意精度的二进制转换，特别是将小数部分转换为二进制表示。希望对有需要的朋友有所帮助。下一篇文章将给出SPIHT的解码程序。请关注后续文章，欢迎 Email 联系交流。
4、精细扫描程序
function Rn=refinement(N,LSP_Old)
% 函数 REFINEMENT()为精细编码程序，对上一级编码产生的重要系数列表LSP_Old，读取每个
% 表项相应小波系数绝对值的二进制表示，输出其中第N个重要的位，即相应于 2^N 处的码数
% 输入参数：N —— 本级编码阈值的指数
% LSP_Old —— 上一级编码产生的重要系数列表
% 输出参数：Rn —— 精细扫描输出位流
global Mat
% Mat是输入的小波分解系数矩阵，作为全局变量，在编码的相关程序中使用
Rn=[];
% 每级精细扫描开始时，Rn 均为空表
% LSP_Old 非空时才执行精细扫描程序
if ~isempty(LSP_Old)
rlsp=size(LSP_Old,1);
% 获取 LSP_Old 的表项个数，对每个表项进行扫描
for r=1:rlsp
tMat=Mat(LSP_Old(r,1),LSP_Old(r,2));
% 读取该表项对应的小波系数值
[biLSP,Np]=fracnum2bin(abs(tMat),N);
% 函数 FRACNUM2BIN() 根据精细扫描对应的权位 N ，将任意的十进制正数转换为二进制数，
% 输出参数为二进制表示列表 biLSP 和 权位N与最高权位的距离 Np 。
Rn=[Rn,biLSP(Np)];
% biLSP(Np)即为小波系数绝对值的二进制表示中第N个重要的位
end
end
（1）十进制数转换为二进制表示的程序
function [binlist,qLpoint]=fracnum2bin(num,qLevel)
% 函数 FRACNUM2BIN() 根据精细扫描对应的权位 N ，将任意的十进制正数转换为二进制数，
% 包括带有任意位小数的十进制数。Matlab中的函数 dec2bin()、dec2binvec()只能将十
% 进制数的整数部分转换为二进制表示，对小数部分则不转换。
%
% 输入参数：num —— 非负的十进制数
% qLevel —— 量化转换精度，也可以是精细扫描对应的权位 N
% 输出参数：biLSP —— 二进制表示列表
% Np —— 权位N与最高权位的距离，N 也是本级编码阈值的指数
intBin=dec2binvec(num);
% 首先用Matlab函数dec2binvec()获取整数部分的二进制表示intBin，低位在前，高位在后
intBin=intBin(end:-1:1);
% 根据个人习惯，将二进制表示转换为高位在前，低位在后
lenIB=length(intBin);
% 求出二进制表示的长度
decpart=num-floor(num);
% 求出小数部分
decBin=[];
% 小数部分的二进制表示初始化为空表
% 根据量化精度要求输出总的二进制表示列表
if (qLevel+1)>lenIB
% 如果量化精度高于整数部分的二进制码长，则输出为零值列表
binlist=zeros(1,qLevel+1);
qLpoint=1;
elseif qLevel>=0
% 如果量化精度在整数权位，则输出整数部分的二进制表示intBin
% 不需转换小数部分，同时输出量化精度与最高权位的距离Np
binlist=intBin;
binlist(lenIB-qLevel+1:end)=0;
qLpoint=lenIB-qLevel;
elseif qLevel<0
% 如果量化精度在小数权位，则需转换小数部分
N=-1;
while N>=qLevel
% 小数部分的转换只需进行到量化精度处
res=decpart-2^N;
if res==0
decBin=[decBin,1];
decBin(end+1:-qLevel)=0;
% 如果小数部分的转换完成时仍未达到量化精度所在的权位，则补零
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
% 输出整数部分和小数部分的二进制表示intBin,decBin，以及量化精度与最高权位的距离Np
end