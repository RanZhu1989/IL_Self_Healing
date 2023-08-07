% 33 Bus Test System Data   V1.0 2023-3-22
% @Matpower "case33bw.m"

%% ================================1. Static Parameters ==========================================
% --------------------------------------------------------- 1.1 Base Value --------------------------
Sb = 100;                % MW/MVar/MVA
Vb = 12.66;             % kV
Zb = Vb^2/Sb;           % O
Ib = Sb/(sqrt(3)*Vb);   % kA
V0 = 1; % p.u.
V_max = 1.05;
V_min = 0.95;
% --------------------------------------------------------- 1.2 Mode --------------------------
NT = 1; % ## 在这里设置步数 step模式为1
% --------------------------------------------------------- 1.3 BigM --------------------------
Big_M_FF = 40; % 单商品流的BigM
Big_M_V = 3; % 压降松弛的BigM
BigM_SC = 2;

% --------------------------------------------------------- 1.4 Constant --------------------------
CP_LC = 10*Sb; %  100$/MW
CP_WC = 1*Sb; % 10$/MW
Pen_IS = min(CP_LC, CP_WC)/1e3; % 为惩罚因子
%% ================================= 2. DN System Data ====================================
DN_Bus_Data=readmatrix('Case_33BW_Data.xlsx','Sheet','Bus_Data'); 
DN_Branch_Data=readmatrix('Case_33BW_Data.xlsx','Sheet','Branch_Data');
Bus_Plot_Data = readmatrix('Case_33BW_Data.xlsx','Sheet','Plot_Data'); % 画图
DG_Data = readmatrix('Case_33BW_Data.xlsx','Sheet','DG_Data'); % dg

N_Line = size(DN_Branch_Data,1);
N_TL = 5; % # TODO 需要增加tieline辨识
N_NL = 32; 
N_Bus = 33;
N_PL = N_Bus - 1;
Pd_Data = DN_Bus_Data(:,2)./Sb;
Pd_Data_all = sum(Pd_Data);
Qd_Data = DN_Bus_Data(:,3)./Sb;
Qd_Data_all = sum(Qd_Data);
Pd_ratio = Pd_Data./Pd_Data_all;    % Active load ratio
Qd_ratio = Qd_Data./Qd_Data_all;    % Reactive load ratio

N_Branch = size(DN_Branch_Data,1);
Branch_start=DN_Branch_Data(:,2); % 每条线路的起点和终点
Branch_end=DN_Branch_Data(:,3);
LS_Mask = MakeMask(N_Bus,N_Branch,Branch_start ); 
LE_Mask = MakeMask(N_Bus,N_Branch,Branch_end ); 
pIn=MakeIncMatrix(Branch_start,Branch_end);  % 潮流方程专用  关联矩阵，node-branch incidence matrix  jk-ij
pInn=pIn;
pInn(pInn>0)=0;    % % 潮流方程专用 Inn is the negative part of I   +ij
R_Branch0 = DN_Branch_Data(:,4)./Zb; % 线路阻抗
X_Branch0 = DN_Branch_Data(:,5)./Zb; % 线路电抗
SZ_Branch0 = R_Branch0.^2+X_Branch0.^2; % SOCP模型专用 线路阻抗模平方
S_Branch0 = DN_Branch_Data(:,6)./Sb;
Alive = DN_Branch_Data(:,7); 

N_DG = size(DG_Data,1);
DataDN.IndDG = DG_Data(:,2); % DG接入位置
DataDN.IndBSDG = DG_Data(find(DG_Data(:,7)==1),2); % DG中黑启动机组
DataDN.IndNMDG = DG_Data(find(DG_Data(:,7)==0),2); % DG中非黑启动机组
N_BSDG = size(DataDN.IndBSDG,1);
N_NMDG = size(DataDN.IndNMDG,1);
P_DG_max0 = DG_Data(:,3)./Sb;
P_DG_min0 = DG_Data(:,4)./Sb;
Q_DG_max0 = DG_Data(:,5)./Sb;
Q_DG_min0 = DG_Data(:,6)./Sb;
DG_Mask = MakeMask(N_Bus, N_DG, DataDN.IndDG); % DG接入关联矩阵
BSDG_Mask = MakeMask(N_Bus, N_BSDG, DataDN.IndBSDG); % BSDG接入关联矩阵
NMDG_Mask = MakeMask(N_Bus, N_NMDG, DataDN.IndNMDG); % Normal DG接入关联矩阵
% 还需要线路损坏状态a，上一时刻线路状态b0，上一时刻电容器状态q0


% 常量后处理，增加时间维度
R_Branch = repmat(R_Branch0, [1, NT]);
X_Branch = repmat(X_Branch0, [1, NT]);
S_Branch = repmat(S_Branch0, [1, NT]);
Pd_all = 3.75 / Sb;
Qd_all = 0.37 / Sb;
Pd = Pd_all.* repmat(Pd_ratio,[1, NT]);
Qd = Qd_all.* repmat(Qd_ratio,[1, NT]);
P_DG_max = repmat(P_DG_max0, [1, NT]);
P_DG_min = repmat(P_DG_min0, [1, NT]);
Q_DG_max = repmat(Q_DG_max0, [1, NT]);
Q_DG_min = repmat(Q_DG_min0, [1, NT]);

% 线路状态
a = repmat(Alive(1:32,:), [1 NT]);  % 初始线路故障
% a([6,11,29,32],:) = 0;  % 线路故障在这里指定 expert和step
a([21,11],:) = 0; 
X_tieline0 = zeros(5,1); % tieline初始状态 expert用
X_tieline_input = [0; 0; 0; 0; 0]; % step 
Q_svc_input = [0.002; 0.002; 0.002; 0.002; 0.002;0.002];
X_rec0 = zeros(33,1);
