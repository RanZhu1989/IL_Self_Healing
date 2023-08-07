% 求解单步 NT = 1
% 需要输入 X_rec0: 上一步负荷pick up情况；X_tieline_input: Tieline控制； Q_svc_input： svc控制

%% 变量

PF = sdpvar(N_Line, NT, 'full'); % active power flow at line ij
QF = sdpvar(N_Line, NT, 'full'); % reactive power flow at line ij
V = sdpvar(N_Bus, NT, 'full'); % voltage at bus j

P_dg = sdpvar(N_DG, NT, 'full'); % 上游
Q_dg = sdpvar(N_DG, NT, 'full');
% delta_Qdg = sdpvar(N_DG, NT-1, 'full');  % ##这个仅在expert中有效，step不用
e_Qsvc = sdpvar(N_DG-1, 1, 'full'); % # step需要指令跟踪误差
e_Qsvc_up = sdpvar(N_DG-1, 1, 'full'); % 用于绝对值
e_Qsvc_down = sdpvar(N_DG-1, 1, 'full');

Pd_rec = sdpvar(N_Bus, NT, 'full');
Qd_rec = sdpvar(N_Bus, NT, 'full');
FF = sdpvar(N_Line, NT,  'full'); % commodity flow at line ij


% binary
X_rec = binvar(N_Bus, NT,  'full'); % load pick up
X_EN = binvar(N_Bus, NT,  'full');
X_tieline = binvar(N_TL, NT,  'full'); % line final state. 对于非TieLine,取决于是否受灾，为常数；对于TieLine则作为变量来控制开关
X_line = binvar(N_NL, NT, 'full'); 

z_bs = binvar(N_Bus, NT,'full'); % 用于MP中两个整数变量X相乘的情况
b = binvar(N_Line, NT, 'full'); % switch state of line ij
sw = sdpvar(N_Line, NT, 'full'); % switch action stage at line ij, from t=2 
X_BS = binvar(N_Bus, NT, 'full'); % 节点是否获得黑启动能力
z_bs1 = binvar(N_Bus, NT, 'full'); % 节点是否黑启动条件判断
z_bs2 = binvar(N_Bus, NT, 'full'); 
% z_dg = binvar(N_DG, NT-1, 'full'); % 稳定SC输出  % ##这个仅在expert中有效，step不用

%% 约束
Cons_DN = [];
% 1. Bus PQ Blance: S_jk - S_ij = S_inj
Cons_DN = [Cons_DN, pIn*PF == DG_Mask*P_dg - Pd_rec ]; 
Cons_DN = [Cons_DN, pIn*QF == DG_Mask*Q_dg -  Qd_rec ]; 

% 2. Voltage : U_j - U_i = r*Q_ij + x*P_ij
Cons_DN = [Cons_DN, pIn'*V-R_Branch.*PF-X_Branch.*QF<=Big_M_V.*(1-b)]; 
Cons_DN = [Cons_DN, pIn'*V-R_Branch.*PF-X_Branch.*QF>=-Big_M_V.*(1-b)]; 
Cons_DN = [Cons_DN, X_BS + X_EN.*V_min - z_bs.*V_min<=V]; 
Cons_DN = [Cons_DN, V <= X_BS + X_EN.*V_max - z_bs.*V_max];
Cons_DN = [Cons_DN, z_bs<=X_BS, z_bs<=X_EN, z_bs>=X_BS+X_EN-1];

% 3. % 3. Load Curtailments
Cons_DN = [Cons_DN, X_rec <= X_EN]; % 
Cons_DN = [Cons_DN, X_rec(1,:) == 0]; % 
Cons_DN = [Cons_DN, Pd_rec == X_rec.*Pd];
Cons_DN = [Cons_DN, Qd_rec == X_rec.*Qd];
% Cons_DN = [Cons_DN, X_rec(:,2:NT) >= X_rec(:,1:NT-1)]; % 恢复的不能失去
Cons_DN = [Cons_DN, X_rec(:,1) >= X_rec0]; % step版本 恢复的不能失去  X_rec0需要从外部输入

% 4. 线路
Cons_DN = [Cons_DN, PF>=-S_Branch.*b]; 
Cons_DN = [Cons_DN, PF<=S_Branch.*b]; 
Cons_DN = [Cons_DN, QF>=-S_Branch.*b]; 
Cons_DN = [Cons_DN, QF<=S_Branch.*b]; 

Cons_Facilities = [];

% 1. DG
Cons_Facilities = [Cons_Facilities, P_dg>=P_DG_min];  % 这里需要再调度一下
Cons_Facilities = [Cons_Facilities, P_dg<=P_DG_max]; 
Cons_Facilities = [Cons_Facilities, Q_dg>=Q_DG_min]; 
Cons_Facilities = [Cons_Facilities, Q_dg<=Q_DG_max]; 
% Step新增
Cons_Facilities = [Cons_Facilities, e_Qsvc == Q_svc_input - Q_dg(2:N_DG,1) ]; % 取除了第一个DG外的无功Q_dg与指令比较
Cons_Facilities = [Cons_Facilities, e_Qsvc == e_Qsvc_up - e_Qsvc_down ]; % 绝对值
Cons_Facilities = [Cons_Facilities, e_Qsvc_up>=0, e_Qsvc_down>=0 ];

% 针对SVC可能存在多解   step不需要
% Cons_Facilities = [Cons_Facilities, BigM_SC*(1-z_dg) <= Q_dg(:,2:NT) - Q_dg(:,1:NT-1) ];
% Cons_Facilities = [Cons_Facilities, -BigM_SC*(1-z_dg) <= delta_Qdg - (Q_dg(:,2:NT) - Q_dg(:,1:NT-1)) ];
% Cons_Facilities = [Cons_Facilities, BigM_SC*(1-z_dg) >= delta_Qdg - (Q_dg(:,2:NT) - Q_dg(:,1:NT-1)) ];
% Cons_Facilities = [Cons_Facilities, -BigM_SC*z_dg <= delta_Qdg + (Q_dg(:,2:NT) - Q_dg(:,1:NT-1)) ];
% Cons_Facilities = [Cons_Facilities, BigM_SC*z_dg >= delta_Qdg + (Q_dg(:,2:NT) - Q_dg(:,1:NT-1)) ];

Cons_Island = [];

% 1. 一个节点为黑启动节点的条件：存在一个BSDG 
Cons_Island = [Cons_Island, X_BS == repmat(sum(BSDG_Mask,2), [1 NT])];

% 2. 每个孤岛是联通的。根据节点是否为黑启动节点，分为两种情况讨论
Cons_Island = [Cons_Island, pIn*FF + X_EN <= Big_M_FF.*(1-z_bs1) ];
Cons_Island = [Cons_Island, pIn*FF + X_EN >= -Big_M_FF.*(1-z_bs1)];
Cons_Island = [Cons_Island, z_bs1 - 1 <= X_BS]; 
Cons_Island = [Cons_Island, X_BS <= 1 - z_bs1];
Cons_Island = [Cons_Island, pIn*FF >= -Big_M_FF.*(1-z_bs2)];
Cons_Island = [Cons_Island, z_bs2 - 1 <= X_BS - 1];
Cons_Island = [Cons_Island, X_BS -1 <= 1 - z_bs2];
Cons_Island = [Cons_Island, X_EN - X_BS >=  -Big_M_FF.*(1-z_bs2)];
Cons_Island = [Cons_Island, X_EN - X_BS <=  Big_M_FF.*(1-z_bs2)];
Cons_Island = [Cons_Island, z_bs1 + z_bs2 == 1];

% 3. 商品流与线路状态
Cons_Island = [Cons_Island, -Big_M_FF.*b<=FF];
Cons_Island = [Cons_Island, FF<=Big_M_FF.*b];
Cons_Island = [Cons_Island, b == [X_line; X_tieline] ]; % b=[Xline; Xtieline]为全体线路状态，X_line是变量，由a决定，a是外部输入的普通线路健康状态
Cons_Island = [Cons_Island, X_line<=a]; % 松弛普通线路

% 4. 闭合的边数=总节点数-带电孤岛数-不带电孤立节点数
Cons_Island = [Cons_Island, sum(b,1) == N_Bus - sum(X_BS,1) - sum(1-X_EN,1) ];

% 线路操作约束
Cons_Island = [Cons_Island, X_tieline == X_tieline_input ]; % 不需要下面的约束，直接从外部输入

% Cons_Island = [Cons_Island, X_tieline(:,2:NT) >= X_tieline(:,1:NT-1) ]; % 闭合的tieline不能打开
% Cons_Island = [Cons_Island, X_tieline(:,1) >= X_tieline0]; % 初值
% Cons_Island = [Cons_Island, sum(X_tieline(:,2:NT)-X_tieline(:,1:NT-1),1) <= 1]; % 每步只能动一个TieLine
% Cons_Island = [Cons_Island, sum(X_tieline(:,1)-X_tieline0,1) <= 1]; % 初值


obj = -sum(Pd_rec,'all')-0.01*sum(X_line,'all')+1*sum(e_Qsvc_up + e_Qsvc_down, 'all');
cons = [Cons_DN, Cons_Facilities, Cons_Island, Q_dg(7,:)==0];
ops = sdpsettings( 'solver', 'gurobi'); 

optimize(cons, obj, ops);
res_v = value(V);
res_b = value(b);
res_Xen = value(X_EN);
res_Xtie = value(X_tieline);
res_Qdg = value(Q_dg);
value(sum(Pd_rec,'all'))