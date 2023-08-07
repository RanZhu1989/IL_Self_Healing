import XLSX

function MakeMask(x, y, list)
    Mask = zeros(x, y)
    for i = 1:x
        Mask[i, findall(x -> x == i, list)] .= 1
    end
    return Mask
end

function MakeIncMatrix(s, t)
    MaxNode = max(maximum(s), maximum(t))
    I = zeros(MaxNode, length(s))
    for j = 1:length(s)
        I[s[j], j] = 1
        I[t[j], j] = -1
    end
    return I
end


Sb = 100                   # MW/MVar/MVA
Vb = 12.66                # kV
Zb = Vb^2/Sb              # O
Ib = Sb/(sqrt(3)*Vb)      # kA
V0 = 1                    # p.u.
V_max = 1.05
V_min = 0.95
NT = 1                     # 在这里设置步数 step模式为1
Big_M_FF = 40             # 单商品流的BigM
Big_M_V = 3               # 压降松弛的BigM
BigM_SC = 2


# 从 Excel 文件读取数据
Bus_Data = XLSX.readtable("Case_33BW_Data.xlsx", "Bus")
Bus_Data = hcat(Bus_Data.data ...)

DG_Data = XLSX.readtable("Case_33BW_Data.xlsx", "DG")
DG_Data = hcat(DG_Data.data ...)

Branch_Data = XLSX.readtable("Case_33BW_Data.xlsx", "Branch")
Branch_Data = hcat(Branch_Data.data ...)


N_Line = size(Branch_Data, 1)
N_TL = 5                    # 需要增加tieline辨识
N_NL = 32 
N_Bus = 33
N_PL = N_Bus - 1
Pd_Data = Bus_Data[:, 2] ./ Sb
Pd_Data_all = sum(Pd_Data)
Qd_Data = Bus_Data[:, 3] ./ Sb
Qd_Data_all = sum(Qd_Data)
Pd_ratio = Pd_Data ./ Pd_Data_all    # Active load ratio
Qd_ratio = Qd_Data ./ Qd_Data_all    # Reactive load ratio

N_Branch = size(Branch_Data, 1)
Branch_start = Branch_Data[:, 2]   # 每条线路的起点和终点
Branch_end = Branch_Data[:, 3]
LS_Mask = MakeMask(N_Bus, N_Branch, Branch_start)
LE_Mask = MakeMask(N_Bus, N_Branch, Branch_end)
pIn = MakeIncMatrix(Branch_start, Branch_end)   # 潮流方程专用  关联矩阵，node-branch incidence matrix  jk-ij
pInn = copy(pIn)
pInn[pInn .> 0] .= 0    # 潮流方程专用 Inn is the negative part of I   +ij
R_Branch0 = Branch_Data[:, 4] ./ Zb   # 线路阻抗
X_Branch0 = Branch_Data[:, 5] ./ Zb   # 线路电抗
SZ_Branch0 = R_Branch0 .^ 2 + X_Branch0 .^ 2   # SOCP模型专用 线路阻抗模平方
S_Branch0 = Branch_Data[:, 6] ./ Sb
Alive = Branch_Data[:, 7] 

N_DG = size(DG_Data, 1)
DataDN_IndDG = DG_Data[:, 2]   # DG接入位置
DataDN_IndBSDG = DG_Data[findall(x -> x == 1, DG_Data[:, 7]), 2]   # DG中黑启动机组
DataDN_IndNMDG = DG_Data[findall(x -> x == 0, DG_Data[:, 7]), 2]   # DG中非黑启动机组
N_BSDG = size(DataDN_IndBSDG, 1)
N_NMDG = size(DataDN_IndNMDG, 1)
P_DG_max0 = DG_Data[:, 3] ./ Sb
P_DG_min0 = DG_Data[:, 4] ./ Sb
Q_DG_max0 = DG_Data[:, 5] ./ Sb
Q_DG_min0 = DG_Data[:, 6] ./ Sb
DG_Mask = MakeMask(N_Bus, N_DG, DataDN_IndDG)   # DG接入关联矩阵
BSDG_Mask = MakeMask(N_Bus, N_BSDG, DataDN_IndBSDG)   # BSDG接入关联矩阵
NMDG_Mask = MakeMask(N_Bus, N_NMDG, DataDN_IndNMDG)   # Normal DG接入关联矩阵


R_Branch = repeat(R_Branch0, 1, NT)
X_Branch = repeat(X_Branch0, 1, NT)
S_Branch = repeat(S_Branch0, 1, NT)
Pd_all = 3.75 / Sb
Qd_all = 0.37 / Sb
Pd = Pd_all .* repeat(Pd_ratio, 1, NT)
Qd = Qd_all .* repeat(Qd_ratio, 1, NT)
P_DG_max = repeat(P_DG_max0, 1, NT)
P_DG_min = repeat(P_DG_min0, 1, NT)
Q_DG_max = repeat(Q_DG_max0, 1, NT)
Q_DG_min = repeat(Q_DG_min0, 1, NT)