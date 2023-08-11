
import pandas as pd
import numpy as np
# import math
from utils_env import *

class System_Data():
    def __init__(self, file_name:str):
        # 初始化只放固定的数据

        V0 = 1.0 # p.u.
        V_max = 1.05
        V_min = 0.95
        Big_M_FF = 40             # 单商品流的BigM
        Big_M_V = 3               # 压降松弛的BigM
        BigM_SC = 2
        Vb = pd.read_excel(file_name,sheet_name='Base').to_numpy().item()       # kV
        Sb = 1.0                # MW/MVar/MVA
        # Ib = Sb/(math.sqrt(3)*Vb)   # kA
        Zb = Vb**2/Sb           # O
        
        Bus_Data=pd.read_excel(file_name,sheet_name='Bus').to_numpy()
        Branch_Data=pd.read_excel(file_name,sheet_name='Branch').to_numpy()
        DG_Data = pd.read_excel(file_name,sheet_name='DG').to_numpy()
        
        N_Bus = Bus_Data.shape[0]
        Pd_Data = Bus_Data[:, 1] / Sb
        Pd_Data_all = np.sum(Pd_Data)
        Qd_Data = Bus_Data[:, 2] / Sb
        Qd_Data_all = np.sum(Qd_Data)
        Pd_ratio = Pd_Data / Pd_Data_all  # Active load ratio
        Qd_ratio = Qd_Data / Qd_Data_all  # Reactive load ratio

        N_Branch = Branch_Data.shape[0]
        N_TL = np.sum(Branch_Data[:, 6] == 1).item()  # 必须加上.item()，否则返回的是array([x]) 导致后面传入julia进程数据类型错误
        NT = N_TL
        N_NL = np.sum(Branch_Data[:, 6] == 0).item()
        Branch_start = Branch_Data[:, 1].astype(int)  #每条线路的起点和终点
        Branch_end = Branch_Data[:, 2].astype(int)
        # LS_Mask = make_mask(N_Bus,N_Branch,Branch_start ) 
        # LE_Mask = make_mask(N_Bus,N_Branch,Branch_end )
        pIn = make_inc_matrix(Branch_start, Branch_end)
        pInn = np.copy(pIn)
        pInn[pInn > 0] = 0
        R_Branch0 = Branch_Data[:, 3] / Zb
        X_Branch0 = Branch_Data[:, 4] / Zb
        # SZ_Branch0 = R_Branch0 ** 2 + X_Branch0 ** 2
        S_Branch0 = Branch_Data[:, 5] / Sb
        
        self.disturbance_set = np.where(Branch_Data[:, 7] == 1)[0].tolist()  # 可能受灾的线路

        N_DG = DG_Data.shape[0]
        DataDN_IndDG = DG_Data[:, 1]
        DataDN_IndBSDG = DG_Data[DG_Data[:, 6] == 1, 1]
        # DataDN_IndNMDG = DG_Data[DG_Data[:, 6] == 0, 1]
        N_BSDG = DataDN_IndBSDG.shape[0]
        # N_NMDG = DataDN_IndNMDG.shape[0]
        P_DG_max0 = DG_Data[:, 2] / Sb
        P_DG_min0 = DG_Data[:, 3] / Sb
        Q_DG_max0 = DG_Data[:, 4] / Sb
        Q_DG_min0 = DG_Data[:, 5] / Sb
        DG_Mask = make_mask(N_Bus, N_DG, DataDN_IndDG)
        BSDG_Mask = make_mask(N_Bus, N_BSDG, DataDN_IndBSDG)
        # NMDG_Mask = make_mask(N_Bus, N_NMDG, DataDN_IndNMDG)

        load_pec = 1.0
        Pd_all = Pd_Data_all * load_pec
        Qd_all = Qd_Data_all * load_pec

        Pd = Pd_all * np.tile(Pd_ratio, (NT, 1)).T
        Qd = Qd_all * np.tile(Qd_ratio, (NT, 1)).T

        R_Branch = np.tile(R_Branch0, (NT, 1)).T
        X_Branch = np.tile(X_Branch0, (NT, 1)).T
        S_Branch = np.tile(S_Branch0, (NT, 1)).T

        P_DG_max = np.tile(P_DG_max0, (NT, 1)).T
        P_DG_min = np.tile(P_DG_min0, (NT, 1)).T
        Q_DG_max = np.tile(Q_DG_max0, (NT, 1)).T
        Q_DG_min = np.tile(Q_DG_min0, (NT, 1)).T
        
        
        self.N_Bus = N_Bus
        self.N_Branch = N_Branch
        self.N_TL = N_TL
        self.NT = NT
        self.N_NL = N_NL
        self.S_lower_limit = -S_Branch0
        self.S_upper_limit = S_Branch0
        self.N_DG = N_DG
        self.Qsvc_lower_limit = Q_DG_min0[1:]
        self.Qsvc_upper_limit = Q_DG_max0[1:]
        self.Pd = Pd
        self.Qd = Qd
        
        
        

        self.args_expert = (NT, N_Branch, N_TL, N_NL, N_Bus, pIn, N_DG, DG_Mask, R_Branch, X_Branch, Big_M_V, V0,
                V_min, V_max, Pd, Qd, S_Branch, P_DG_min, P_DG_max, Q_DG_min, Q_DG_max, BigM_SC, BSDG_Mask,
                Big_M_FF)

        self.args_step = (1, N_Branch, N_TL, N_NL, N_Bus, pIn, N_DG, DG_Mask, R_Branch[:, 0], X_Branch[:, 0], Big_M_V, V0,
                V_min, V_max, Pd[:, 0], Qd[:, 0], S_Branch[:, 0], P_DG_min[:, 0], P_DG_max[:, 0], Q_DG_min[:, 0],
                Q_DG_max[:, 0], BSDG_Mask, Big_M_FF)
