import numpy as np
import gurobipy as gp
from gurobipy import GRB
from typing import Optional, Tuple 

class OPF_Core:
    
    def __init__(self, args_expert:tuple, args_step:tuple, 
                 MIP_gap_expert_model:float=1e-4, MIP_gap_step_model:float=1e-4,
                 MIP_gap_reset_model:float=1e-4, display:bool=True) -> None:
        
        self.expert_model = self.make_expert_model(args_expert)
        self.step_model = self.make_step_model(args_step)
        self.reset_model = self.make_reset_model(args_step)
        
        self.expert_model.Params.OutputFlag, self.step_model.Params.OutputFlag, \
            self.reset_model.Params.OutputFlag = display, display, display
            
        self.expert_model.setParam('MIPGap', MIP_gap_expert_model)
        self.step_model.setParam('MIPGap', MIP_gap_step_model)
        self.reset_model.setParam('MIPGap', MIP_gap_reset_model)
        pass
    
    def make_expert_model(self, args_expert:tuple) -> gp.Model:
        
        NT, N_Branch, N_TL, N_NL, N_Bus, pIn, N_DG, DG_Mask, R_Branch, X_Branch, Big_M_V, V0, \
            V_min, V_max, Pd, Qd, S_Branch, P_DG_min, P_DG_max, Q_DG_min, Q_DG_max, BigM_SC, BSDG_Mask, \
            Big_M_FF = args_expert
        
        self.N_DG = N_DG
        self.NT = NT
        self.N_Bus = N_Bus
        self.N_Branch = N_Branch
        self.N_TL = N_TL
        self.N_NL = N_NL
            
        expert_model = gp.Model("Expert_Model")
        
        # 注意使用mvar的时候，需要注意lb默认为0，需要修改
        a = expert_model.addMVar(shape=(N_NL,NT), lb=float("-inf"), vtype=GRB.BINARY, name="a") #NOTE 二维
        X_tieline0 = expert_model.addMVar(shape=N_TL, lb=float("-inf"), vtype=GRB.BINARY, name="X_tieline0") #NOTE 一维
        Q_svc = expert_model.addMVar(shape=(N_DG-1,NT), lb=float("-inf"), vtype=GRB.CONTINUOUS, name="Q_svc")
        X_rec0 = expert_model.addMVar(shape=N_Bus, lb=float("-inf"), vtype=GRB.BINARY, name="X_rec0")
        X_line0 = expert_model.addMVar(shape=N_NL, lb=float("-inf"), vtype=GRB.BINARY, name="X_line0")
        
        PF = expert_model.addMVar(shape=(N_Branch,NT), lb=float("-inf"), vtype=GRB.CONTINUOUS, name="PF")
        QF = expert_model.addMVar(shape=(N_Branch,NT), lb=float("-inf"), vtype=GRB.CONTINUOUS, name="QF")
        V = expert_model.addMVar(shape=(N_Bus,NT), lb=float("-inf"), vtype=GRB.CONTINUOUS, name="V")

        P_dg = expert_model.addMVar(shape=(N_DG,NT), lb=float("-inf"), vtype=GRB.CONTINUOUS, name="P_dg")
        Q_dg = expert_model.addMVar(shape=(N_DG,NT), lb=float("-inf"), vtype=GRB.CONTINUOUS, name="Q_dg")
        delta_Qdg = expert_model.addMVar(shape=(N_DG-1,NT-1), lb=float("-inf"), vtype=GRB.CONTINUOUS, name="delta_dg")

        Pd_rec = expert_model.addMVar(shape=(N_Bus,NT), lb=float("-inf"), vtype=GRB.CONTINUOUS, name="Pd_rec")
        Qd_rec = expert_model.addMVar(shape=(N_Bus,NT), lb=float("-inf"), vtype=GRB.CONTINUOUS, name="Qd_rec")
        FF = expert_model.addMVar(shape=(N_Branch,NT), lb=float("-inf"), vtype=GRB.CONTINUOUS, name="FF")

        X_rec = expert_model.addMVar(shape=(N_Bus,NT), lb=float("-inf"), vtype=GRB.BINARY, name="X_rec")
        X_EN = expert_model.addMVar(shape=(N_Bus,NT), lb=float("-inf"), vtype=GRB.BINARY, name="X_EN")
        X_tieline = expert_model.addMVar(shape=(N_TL,NT), lb=float("-inf"), vtype=GRB.BINARY, name="X_tieline")
        X_line = expert_model.addMVar(shape=(N_NL,NT), lb=float("-inf"), vtype=GRB.BINARY, name="X_line")

        z_bs = expert_model.addMVar(shape=(N_Bus,NT), lb=float("-inf"), vtype=GRB.BINARY, name="z_bs")
        b = expert_model.addMVar(shape=(N_Branch,NT), lb=float("-inf"), vtype=GRB.BINARY, name="b")
        X_BS = expert_model.addMVar(shape=(N_Bus,NT), lb=float("-inf"), vtype=GRB.BINARY, name="X_BS")
        z_bs1 = expert_model.addMVar(shape=(N_Bus,NT), lb=float("-inf"), vtype=GRB.BINARY, name="z_bs1")
        z_bs2 = expert_model.addMVar(shape=(N_Bus,NT), lb=float("-inf"), vtype=GRB.BINARY, name="z_bs2")
        z_dg = expert_model.addMVar(shape=(N_DG-1,NT-1), lb=float("-inf"), vtype=GRB.BINARY, name="z_dg")
        
        
         # ------------------潮流--------------------
            # 1. Bus PQ Blance: S_jk - S_ij = S_inj
        expert_model.addConstr(pIn @ PF == DG_Mask @ P_dg - Pd_rec)
        expert_model.addConstr(pIn @ QF == DG_Mask @ Q_dg - Qd_rec)

        # 2. Voltage : U_j - U_i = r*Q_ij + x*P_ij
        expert_model.addConstr(pIn.T @ V - R_Branch * PF - X_Branch * QF <= Big_M_V * (1 - b))
        expert_model.addConstr(pIn.T @ V - R_Branch * PF - X_Branch * QF >= -Big_M_V * (1 - b))
        expert_model.addConstr(X_BS + V_min * X_EN - V_min * z_bs <= V)
        expert_model.addConstr(V <= V0 * X_BS + V_max * X_EN - V_max * z_bs)
        expert_model.addConstr(z_bs <= X_BS)
        expert_model.addConstr(z_bs <= X_EN)
        expert_model.addConstr(z_bs >= X_BS + X_EN - 1)
        # 3. % 3. Load Curtailments
        expert_model.addConstr(X_rec <= X_EN)
        expert_model.addConstr(X_rec[0,:] == 0)
        expert_model.addConstr(Pd_rec == X_rec * Pd)
        expert_model.addConstr(Qd_rec == X_rec * Qd)
        expert_model.addConstr(X_rec[:,1:] >= X_rec[:,0:-1])
        expert_model.addConstr(X_rec[:,0] >= X_rec0)
            # % 4. 线路
        expert_model.addConstr(PF >= -S_Branch * b)
        expert_model.addConstr(PF <= S_Branch * b)
        expert_model.addConstr(QF >= -S_Branch * b)
        expert_model.addConstr(QF <= S_Branch * b)
        # ------------DG ----------------
        expert_model.addConstr(P_dg >= (DG_Mask.T @ X_EN) * P_DG_min)
        expert_model.addConstr(P_dg <= (DG_Mask.T @ X_EN) * P_DG_max)
        expert_model.addConstr(Q_dg >= (DG_Mask.T @ X_EN) * Q_DG_min)
        expert_model.addConstr(Q_dg <= (DG_Mask.T @ X_EN) * Q_DG_max)
        expert_model.addConstr(Q_dg[1:,:] == Q_svc) # Q_svc定义
        
        # 由于是固定时间断面，针对SVC可能存在多解
        expert_model.addConstr(BigM_SC * (1 - z_dg) <= Q_dg[1:,1:] - Q_dg[1:,0:-1])
        expert_model.addConstr(-BigM_SC * (1 - z_dg) <= delta_Qdg - (Q_dg[1:,1:] - Q_dg[1:,0:-1]) )
        expert_model.addConstr(BigM_SC * (1 - z_dg) >= delta_Qdg - (Q_dg[1:,1:] - Q_dg[1:,0:-1]) )
        expert_model.addConstr(-BigM_SC * z_dg <= delta_Qdg + (Q_dg[1:,1:] - Q_dg[1:,0:-1]) )
        expert_model.addConstr(BigM_SC * z_dg >= delta_Qdg + (Q_dg[1:,1:] - Q_dg[1:,0:-1]) )

        # ---------------Island----------------
            #  1. 一个节点为黑启动节点的条件：存在一个BSDG 
        expert_model.addConstr(X_BS == BSDG_Mask.sum(axis=1, keepdims=True) * np.ones((1, NT)) )
            # % 2. 每个孤岛是联通的。根据节点是否为黑启动节点，分为两种情况讨论
        expert_model.addConstr(pIn @ FF + X_EN <= Big_M_FF * (1 - z_bs1))
        expert_model.addConstr(pIn @ FF + X_EN >= -Big_M_FF * (1 - z_bs1))
        expert_model.addConstr(z_bs1 - 1 <= X_BS)
        expert_model.addConstr(X_BS <= 1 - z_bs1)
        expert_model.addConstr(pIn @ FF >= -Big_M_FF * (1 - z_bs2))
        expert_model.addConstr(z_bs2 - 1 <= X_BS - 1)
        expert_model.addConstr(X_BS - 1 <= 1 - z_bs2)
        expert_model.addConstr(X_EN - X_BS >= -Big_M_FF * (1 - z_bs2))
        expert_model.addConstr(X_EN - X_BS <= Big_M_FF * (1 - z_bs2))
        expert_model.addConstr(z_bs1 + z_bs2 == 1 )

        # % 3. 商品流与线路状态
        expert_model.addConstr(-Big_M_FF * b <= FF)
        expert_model.addConstr(FF <= Big_M_FF * b)
        expert_model.addConstr(b[0:N_NL,:] == X_line) # b=[Xline; Xtieline]为全体线路状态，X_line是变量，由a决定，a是外部输入的普通线路健康状态
        expert_model.addConstr(b[N_NL:,:] == X_tieline) # X_tieline是变量，由a决定，a是外部输入的跨区线路健康状态
        expert_model.addConstr(X_line <= a ) #NOTE a 需要在外部输入 这个仅在env.rest才需要

        #  4. 闭合的边数=总节点数-带电孤岛数-不带电孤立节点数
        expert_model.addConstr(b.sum(axis=0) == N_Bus - X_BS.sum(axis=0) - (1 - X_EN).sum(axis=0))


            # % 线路操作约束
        expert_model.addConstr(X_tieline[:, 1:] >= X_tieline[:, 0:-1])

        expert_model.addConstr(X_tieline[:, 0] >= X_tieline0) #NOTE X_tieline0 需要在外部输入
        expert_model.addConstr( (X_tieline[:, 1:] - X_tieline[:, 0:-1]).sum(axis=0) <= 1)
        expert_model.addConstr( (X_tieline[:, 0] - X_tieline0).sum(axis=0) <= 1) #NOTE X_tieline0 需要在外部输入
        expert_model.addConstr( X_line[:, 1:] >= X_line[:, 0:-1])
        expert_model.addConstr( X_line[:, 0] >= X_line0)

        # objective
        expert_model.setObjective(-Pd_rec.sum() - 0.01 * X_line.sum() + 1 * delta_Qdg.sum(), GRB.MINIMIZE)
        
        expert_model.update() #NOTE !! This is important
                
        return expert_model
    
    def make_step_model(self, args_step:tuple) -> gp.Model:
        
        _, N_Branch, N_TL, N_NL, N_Bus, pIn, N_DG, DG_Mask, R_Branch, X_Branch, Big_M_V, V0, \
            V_min, V_max, Pd, Qd, S_Branch, P_DG_min, P_DG_max, Q_DG_min, \
            Q_DG_max, BSDG_Mask, Big_M_FF = args_step
        
        step_model = gp.Model("Step_Model")
        
        a = step_model.addMVar(shape=N_NL, lb=float("-inf"), vtype=GRB.BINARY, name="a") #NOTE 这些都是一维
        X_rec0 = step_model.addMVar(shape=N_Bus, lb=float("-inf"), vtype=GRB.BINARY, name="X_rec0")
        Q_svc = step_model.addMVar(shape=N_DG-1, lb=float("-inf"), vtype=GRB.CONTINUOUS, name="Q_svc")
        
        PF = step_model.addMVar(shape=N_Branch, lb=float("-inf"), vtype=GRB.CONTINUOUS, name="PF")
        QF = step_model.addMVar(shape=N_Branch, lb=float("-inf"), vtype=GRB.CONTINUOUS, name="QF")
        V = step_model.addMVar(shape=N_Bus, lb=float("-inf"), vtype=GRB.CONTINUOUS, name="V")
        
        P_dg = step_model.addMVar(shape=N_DG, lb=float("-inf"), vtype=GRB.CONTINUOUS, name="P_dg")
        Q_dg = step_model.addMVar(shape=N_DG, lb=float("-inf"), vtype=GRB.CONTINUOUS, name="Q_dg")
        e_Qsvc = step_model.addMVar(shape=N_DG-1, lb=float("-inf"), vtype=GRB.CONTINUOUS, name="e_Qsvc")
        e_Qsvc_up = step_model.addMVar(shape=N_DG-1, lb=float("-inf"), vtype=GRB.CONTINUOUS, name="e_Qsvc_up")
        e_Qvsc_down = step_model.addMVar(shape=N_DG-1, lb=float("-inf"), vtype=GRB.CONTINUOUS, name="e_Qsvc_down")
        
        Pd_rec = step_model.addMVar(shape=N_Bus, lb=float("-inf"), vtype=GRB.CONTINUOUS, name="Pd_rec")
        Qd_rec = step_model.addMVar(shape=N_Bus, lb=float("-inf"), vtype=GRB.CONTINUOUS, name="Qd_rec")
        FF = step_model.addMVar(shape=N_Branch, lb=float("-inf"), vtype=GRB.CONTINUOUS, name="FF")
        
        X_rec = step_model.addMVar(shape=N_Bus, lb=float("-inf"), vtype=GRB.BINARY, name="X_rec")
        X_EN = step_model.addMVar(shape=N_Bus, lb=float("-inf"), vtype=GRB.BINARY, name="X_EN")
        X_tieline = step_model.addMVar(shape=N_TL, lb=float("-inf"), vtype=GRB.BINARY, name="X_tieline")
        X_line = step_model.addMVar(shape=N_NL, lb=float("-inf"), vtype=GRB.BINARY, name="X_line")
        
        z_bs = step_model.addMVar(shape=N_Bus, lb=float("-inf"), vtype=GRB.BINARY, name="z_bs")
        b = step_model.addMVar(shape=N_Branch, lb=float("-inf"), vtype=GRB.BINARY, name="b")
        X_BS = step_model.addMVar(shape=N_Bus, lb=float("-inf"), vtype=GRB.BINARY, name="X_BS")
        z_bs1 = step_model.addMVar(shape=N_Bus, lb=float("-inf"), vtype=GRB.BINARY, name="z_bs1")
        z_bs2 = step_model.addMVar(shape=N_Bus, lb=float("-inf"), vtype=GRB.BINARY, name="z_bs2")
        
         # ------------------潮流--------------------
            # 1. Bus PQ Blance: S_jk - S_ij = S_inj
        step_model.addConstr(pIn @ PF == DG_Mask @ P_dg - Pd_rec)
        step_model.addConstr(pIn @ QF == DG_Mask @ Q_dg - Qd_rec)

        # 2. Voltage : U_j - U_i = r*Q_ij + x*P_ij
        step_model.addConstr(pIn.T @ V - R_Branch * PF - X_Branch * QF <= Big_M_V * (1 - b))
        step_model.addConstr(pIn.T @ V - R_Branch * PF - X_Branch * QF >= -Big_M_V * (1 - b))
        step_model.addConstr(V0 * X_BS + V_min * X_EN - V_min * z_bs <= V)
        step_model.addConstr(V <= V0 * X_BS + V_max * X_EN - V_max * z_bs)
        step_model.addConstr(z_bs <= X_BS)
        step_model.addConstr(z_bs <= X_EN)
        step_model.addConstr(z_bs >= X_BS + X_EN - 1)
        
        # 3. % 3. Load Curtailments
        step_model.addConstr(X_rec <= X_EN)
        step_model.addConstr(X_rec[0] == 0)
        step_model.addConstr(Pd_rec == X_rec * Pd)
        step_model.addConstr(Qd_rec == X_rec * Qd)
        step_model.addConstr(X_rec >= X_rec0)
        
        # % 4. 线路
        step_model.addConstr(PF >= -S_Branch * b)
        step_model.addConstr(PF <= S_Branch * b)
        step_model.addConstr(QF >= -S_Branch * b)
        step_model.addConstr(QF <= S_Branch * b)
        
        # ------------DG ----------------
        step_model.addConstr(P_dg >= (DG_Mask.T @ X_EN) * P_DG_min)
        step_model.addConstr(P_dg <= (DG_Mask.T @ X_EN) * P_DG_max)
        step_model.addConstr(Q_dg >= (DG_Mask.T @ X_EN) * Q_DG_min)
        step_model.addConstr(Q_dg <= (DG_Mask.T @ X_EN) * Q_DG_max)
        
        step_model.addConstr(e_Qsvc == Q_svc - Q_dg[1:])
        step_model.addConstr(e_Qsvc == e_Qsvc_up - e_Qvsc_down)
        step_model.addConstr(e_Qsvc_up >= 0)
        step_model.addConstr(e_Qvsc_down >= 0)
        
        # ---------------Island----------------
            #  1. 一个节点为黑启动节点的条件：存在一个BSDG 
        step_model.addConstr(X_BS == BSDG_Mask.sum(axis=1) )
            # % 2. 每个孤岛是联通的。根据节点是否为黑启动节点，分为两种情况讨论
        step_model.addConstr(pIn @ FF + X_EN <= Big_M_FF * (1 - z_bs1))
        step_model.addConstr(pIn @ FF + X_EN >= -Big_M_FF * (1 - z_bs1))
        step_model.addConstr(z_bs1 - 1 <= X_BS)
        step_model.addConstr(X_BS <= 1 - z_bs1)
        step_model.addConstr(pIn @ FF >= -Big_M_FF * (1 - z_bs2))
        step_model.addConstr(z_bs2 - 1 <= X_BS - 1)
        step_model.addConstr(X_BS - 1 <= 1 - z_bs2)
        step_model.addConstr(X_EN - X_BS >= -Big_M_FF * (1 - z_bs2))
        step_model.addConstr(X_EN - X_BS <= Big_M_FF * (1 - z_bs2))
        step_model.addConstr(z_bs1 + z_bs2 == 1 )

        # % 3. 商品流与线路状态
        step_model.addConstr(-Big_M_FF * b <= FF)
        step_model.addConstr(FF <= Big_M_FF * b)
        step_model.addConstr(b[0:N_NL] == X_line) # b=[Xline; Xtieline]为全体线路状态，X_line是变量，由a决定，a是外部输入的普通线路健康状态
        step_model.addConstr(b[N_NL:] == X_tieline) # X_tieline是变量，由a决定，a是外部输入的跨区线路健康状态
        step_model.addConstr(X_line <= a ) #NOTE a 需要在外部输入 这个仅在env.rest才需要
        
        #  4. 闭合的边数=总节点数-带电孤岛数-不带电孤立节点数
        step_model.addConstr(b.sum(axis=0) == N_Bus - X_BS.sum(axis=0) - (1 - X_EN).sum(axis=0))
        
        # objective
        step_model.setObjective(-Pd_rec.sum() - 0.01 * X_line.sum() + 1000 * (e_Qsvc_up + e_Qvsc_down).sum(), GRB.MINIMIZE)
        
        step_model.update() #NOTE !! This is important
        
        return step_model
    
    def make_reset_model(self,args_step:tuple) -> gp.Model:
        
        _, N_Branch, N_TL, N_NL, N_Bus, pIn, N_DG, DG_Mask, R_Branch, X_Branch, Big_M_V, V0, \
            V_min, V_max, Pd, Qd, S_Branch, P_DG_min, P_DG_max, Q_DG_min, \
            Q_DG_max, BSDG_Mask, Big_M_FF = args_step
        
        reset_model = gp.Model("Reset_Model")
        
        a = reset_model.addMVar(shape=N_NL, lb=float("-inf"), vtype=GRB.BINARY, name="a")  #NOTE 这些都是一维
        Q_svc = reset_model.addMVar(shape=N_DG-1, lb=float("-inf"), vtype=GRB.CONTINUOUS, name="Q_svc")
                
        PF = reset_model.addMVar(shape=N_Branch, lb=float("-inf"), vtype=GRB.CONTINUOUS, name="PF")
        QF = reset_model.addMVar(shape=N_Branch, lb=float("-inf"), vtype=GRB.CONTINUOUS, name="QF")
        V = reset_model.addMVar(shape=N_Bus, lb=float("-inf"), vtype=GRB.CONTINUOUS, name="V")
        
        P_dg = reset_model.addMVar(shape=N_DG, lb=float("-inf"), vtype=GRB.CONTINUOUS, name="P_dg")
        Q_dg = reset_model.addMVar(shape=N_DG, lb=float("-inf"), vtype=GRB.CONTINUOUS, name="Q_dg")
        
        Pd_rec = reset_model.addMVar(shape=N_Bus, lb=float("-inf"), vtype=GRB.CONTINUOUS, name="Pd_rec")
        Qd_rec = reset_model.addMVar(shape=N_Bus, lb=float("-inf"), vtype=GRB.CONTINUOUS, name="Qd_rec")
        FF = reset_model.addMVar(shape=N_Branch, lb=float("-inf"), vtype=GRB.CONTINUOUS, name="FF")
        
        X_rec = reset_model.addMVar(shape=N_Bus, lb=float("-inf"), vtype=GRB.BINARY, name="X_rec")
        X_EN = reset_model.addMVar(shape=N_Bus, lb=float("-inf"), vtype=GRB.BINARY, name="X_EN")
        X_tieline = reset_model.addMVar(shape=N_TL, lb=float("-inf"), vtype=GRB.BINARY, name="X_tieline")
        X_line = reset_model.addMVar(shape=N_NL, lb=float("-inf"), vtype=GRB.BINARY, name="X_line")
          
        z_bs = reset_model.addMVar(shape=N_Bus, lb=float("-inf"), vtype=GRB.BINARY, name="z_bs")
        b = reset_model.addMVar(shape=N_Branch, lb=float("-inf"), vtype=GRB.BINARY, name="b")
        X_BS = reset_model.addMVar(shape=N_Bus, lb=float("-inf"), vtype=GRB.BINARY, name="X_BS")
        z_bs1 = reset_model.addMVar(shape=N_Bus, lb=float("-inf"), vtype=GRB.BINARY, name="z_bs1")
        z_bs2 = reset_model.addMVar(shape=N_Bus, lb=float("-inf"), vtype=GRB.BINARY, name="z_bs2")
        
         # ------------------潮流--------------------
            # 1. Bus PQ Blance: S_jk - S_ij = S_inj
        reset_model.addConstr(pIn @ PF == DG_Mask @ P_dg - Pd_rec)
        reset_model.addConstr(pIn @ QF == DG_Mask @ Q_dg - Qd_rec)

        # 2. Voltage : U_j - U_i = r*Q_ij + x*P_ij
        reset_model.addConstr(pIn.T @ V - R_Branch * PF - X_Branch * QF <= Big_M_V * (1 - b))
        reset_model.addConstr(pIn.T @ V - R_Branch * PF - X_Branch * QF >= -Big_M_V * (1 - b))
        reset_model.addConstr(V0 * X_BS + V_min * X_EN - V_min * z_bs <= V)
        reset_model.addConstr(V <= V0 * X_BS + V_max * X_EN - V_max * z_bs)
        reset_model.addConstr(z_bs <= X_BS)
        reset_model.addConstr(z_bs <= X_EN)
        reset_model.addConstr(z_bs >= X_BS + X_EN - 1)
        
        # 3. % 3. Load Curtailments
        reset_model.addConstr(X_rec <= X_EN)
        reset_model.addConstr(X_rec[0] == 0)
        reset_model.addConstr(Pd_rec == X_rec * Pd)
        reset_model.addConstr(Qd_rec == X_rec * Qd)
        
        # % 4. 线路
        reset_model.addConstr(PF >= -S_Branch * b)
        reset_model.addConstr(PF <= S_Branch * b)
        reset_model.addConstr(QF >= -S_Branch * b)
        reset_model.addConstr(QF <= S_Branch * b)
        
        # ------------DG ----------------
        reset_model.addConstr(P_dg >= (DG_Mask.T @ X_EN) * P_DG_min)
        reset_model.addConstr(P_dg <= (DG_Mask.T @ X_EN) * P_DG_max)
        reset_model.addConstr(Q_dg >= (DG_Mask.T @ X_EN) * Q_DG_min)
        reset_model.addConstr(Q_dg <= (DG_Mask.T @ X_EN) * Q_DG_max)
        reset_model.addConstr(Q_svc == Q_dg[1:])

        
        # ---------------Island----------------
            #  1. 一个节点为黑启动节点的条件：存在一个BSDG 
        reset_model.addConstr(X_BS == BSDG_Mask.sum(axis=1) )
            # % 2. 每个孤岛是联通的。根据节点是否为黑启动节点，分为两种情况讨论
        reset_model.addConstr(pIn @ FF + X_EN <= Big_M_FF * (1 - z_bs1))
        reset_model.addConstr(pIn @ FF + X_EN >= -Big_M_FF * (1 - z_bs1))
        reset_model.addConstr(z_bs1 - 1 <= X_BS)
        reset_model.addConstr(X_BS <= 1 - z_bs1)
        reset_model.addConstr(pIn @ FF >= -Big_M_FF * (1 - z_bs2))
        reset_model.addConstr(z_bs2 - 1 <= X_BS - 1)
        reset_model.addConstr(X_BS - 1 <= 1 - z_bs2)
        reset_model.addConstr(X_EN - X_BS >= -Big_M_FF * (1 - z_bs2))
        reset_model.addConstr(X_EN - X_BS <= Big_M_FF * (1 - z_bs2))
        reset_model.addConstr(z_bs1 + z_bs2 == 1 )

        # % 3. 商品流与线路状态
        reset_model.addConstr(-Big_M_FF * b <= FF)
        reset_model.addConstr(FF <= Big_M_FF * b)
        reset_model.addConstr(b[0:N_NL] == X_line) # b=[Xline; Xtieline]为全体线路状态，X_line是变量，由a决定，a是外部输入的普通线路健康状态
        reset_model.addConstr(b[N_NL:] == X_tieline) # X_tieline是变量，由a决定，a是外部输入的跨区线路健康状态
        reset_model.addConstr(X_line <= a) #NOTE a 需要在外部输入 这个仅在env.rest才需要
        
        #  4. 闭合的边数=总节点数-带电孤岛数-不带电孤立节点数
        reset_model.addConstr(b.sum(axis=0) == N_Bus - X_BS.sum(axis=0) - (1 - X_EN).sum(axis=0))
        
        # objective
        reset_model.setObjective(-Pd_rec.sum() - 0.01 * X_line.sum() , GRB.MINIMIZE)
        
        reset_model.update() #NOTE !! This is important
        
        return reset_model
    
    def set_dmg(self, a_input:np.ndarray) -> None:
        # reset后，为三个模型设置普通线路状态a
        _fixMvar(model=self.expert_model, mvar_name="a",shape=list(a_input.shape), value=a_input, cons_name="fix_a")      
        _fixMvar(model=self.step_model, mvar_name="a",shape=[a_input.shape[0]], value=a_input[:,0], cons_name="fix_a") 
        _fixMvar(model=self.reset_model, mvar_name="a",shape=[a_input.shape[0]], value=a_input[:,0], cons_name="fix_a")
        
        
    def set_ExpertModel(self, X_tieline0_input:np.ndarray, X_rec0_input:np.ndarray, X_line0_input:np.ndarray, vvo:bool=True) -> None:
        # 为ExpertModel设置TieLine初始状态X_tieline0
        _fixMvar(model=self.expert_model, 
                    mvar_name="X_tieline0",shape=list(X_tieline0_input.shape), 
                    value=X_tieline0_input, cons_name="fix_X_tieline")
        # 负荷pick up状态由reset model决定
        _fixMvar(model=self.expert_model,
                    mvar_name="X_rec0",shape=list(X_rec0_input.shape),
                    value=X_rec0_input, cons_name="fix_X_rec0")
        # 需要由reset model确定普通线路初始状态
        _fixMvar(model=self.expert_model,
                    mvar_name="X_line0",shape=list(X_line0_input.shape),
                    value=X_line0_input, cons_name="fix_X_line")
        # 如果不启用VVO，则需要强制SVC输出为0
        if not vvo:
            _fixMvar(model=self.expert_model, 
                        mvar_name="Q_svc",shape=[self.N_DG-1,self.NT], 
                        value=np.zeros((self.N_DG-1,self.NT)), cons_name="fix_Q_svc")
                    
        pass

    def set_StepModel(self, X_rec0_input:np.ndarray, X_tieline_input:np.ndarray,
                      Q_svc_input:Optional[np.ndarray]=None, vvo:bool=True) -> None:
        
        if vvo & (Q_svc_input==None):
            raise ValueError("Please provide a value for Q_svc_input when vvo mode is set to true.")
        
        _fixMvar(model=self.step_model, 
                    mvar_name="X_rec0",shape=list(X_rec0_input.shape),
                    value=X_rec0_input, cons_name="fix_X_rec0")
        
        _fixMvar(model=self.step_model,
                    mvar_name="X_tieline",shape=list(X_tieline_input.shape),
                    value=X_tieline_input, cons_name="fix_X_tieline")
        
        # 是否考虑VVO的区别在于是否接受Q_svc的输入
        if vvo:
            _fixMvar(model=self.step_model,
                        mvar_name="Q_svc",shape=list(Q_svc_input.shape),
                        value=Q_svc_input, cons_name="fix_Q_svc")
        else:
            _fixMvar(model=self.step_model, 
                        mvar_name="Q_svc",shape=[self.N_DG-1], 
                        value=np.zeros(self.N_DG-1), cons_name="fix_Q_svc")
                
        pass
    

    def set_ResetModel(self, X_tieline_input:np.ndarray, Q_svc_input:np.ndarray) -> None:
        
        _fixMvar(model=self.reset_model,
                    mvar_name="X_tieline",shape=list(X_tieline_input.shape),
                    value=X_tieline_input, cons_name="fix_X_tieline")
        
        _fixMvar(model=self.reset_model,
                    mvar_name="Q_svc",shape=list(Q_svc_input.shape),
                    value=Q_svc_input, cons_name="fix_Q_svc")
                
        pass

    def solve_ExpertModel(self) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], 
                                         Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        
        self.expert_model.optimize()
        solved_flag = False
        if self.expert_model.status == GRB.OPTIMAL:
            solved_flag = True
            b = _getX_MvarByName(model=self.expert_model, mvar_name="b", shape=[self.N_Branch,self.NT])
            x_tieline = _getX_MvarByName(model=self.expert_model, mvar_name="X_tieline", shape=[self.N_TL,self.NT])
            x_load = _getX_MvarByName(model=self.expert_model, mvar_name="X_rec", shape=[self.N_Bus,self.NT])
            PF = _getX_MvarByName(model=self.expert_model, mvar_name="PF", shape=[self.N_Branch,self.NT])
            QF = _getX_MvarByName(model=self.expert_model, mvar_name="QF", shape=[self.N_Branch,self.NT])
            Prec = np.sum(_getX_MvarByName(model=self.expert_model, mvar_name="Pd_rec", shape=[self.N_Bus,self.NT]), axis=0)
        
            return solved_flag, b, x_tieline, x_load, PF, QF, Prec
        else:
            return solved_flag, None, None, None, None, None, None
    
    
    def solve_StepModel(self) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray], 
                                       Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray],
                                       Optional[np.ndarray], Optional[np.ndarray] ]:
        
        self.step_model.optimize()
        
        solved_flag = False
        if self.step_model.status == GRB.OPTIMAL:
            solved_flag = True
            b = _getX_MvarByName(model=self.step_model, mvar_name="b", shape=[self.N_Branch])
            x_tieline = _getX_MvarByName(model=self.step_model, mvar_name="X_tieline", shape=[self.N_TL])
            x_load = _getX_MvarByName(model=self.step_model, mvar_name="X_rec", shape=[self.N_Bus])
            PF = _getX_MvarByName(model=self.step_model, mvar_name="PF", shape=[self.N_Branch])
            QF = _getX_MvarByName(model=self.step_model, mvar_name="QF", shape=[self.N_Branch])
            Prec = np.sum(_getX_MvarByName(model=self.step_model, mvar_name="Pd_rec", shape=[self.N_Bus]))
            e_Qsvc = np.sum(_getX_MvarByName(model=self.step_model, mvar_name="e_Qsvc_up", shape=[self.N_DG-1])
                            + _getX_MvarByName(model=self.step_model, mvar_name="e_Qsvc_down", shape=[self.N_DG-1]))
            
            return solved_flag, b, x_tieline, x_load, PF, QF, Prec, e_Qsvc
        
        else:
            
            return solved_flag, None, None, None, None, None, None, None
            
    
    def solve_ResetModel(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        
        self.reset_model.optimize()
        
        b = _getX_MvarByName(model=self.reset_model, mvar_name="b", shape=[self.N_Branch])
        x_tieline = _getX_MvarByName(model=self.reset_model, mvar_name="X_tieline", shape=[self.N_TL])
        x_load = _getX_MvarByName(model=self.reset_model, mvar_name="X_rec", shape=[self.N_Bus])
        PF = _getX_MvarByName(model=self.reset_model, mvar_name="PF", shape=[self.N_Branch])
        QF = _getX_MvarByName(model=self.reset_model, mvar_name="QF", shape=[self.N_Branch])
        Prec = np.sum(_getX_MvarByName(model=self.reset_model, mvar_name="Pd_rec", shape=[self.N_Bus]))
        
        return b, x_tieline, x_load, PF, QF, Prec
    
"""-------------------------------------------------------------------------------------------------------------------"""
# utils for gurobi v10 model

def _getMvarByName(model:gp.Model,mvar_name:str,shape:list) -> dict:
    """Mar version of getVarByName in gurobi 10.0

    Args:
        model (gp.Model): gurobi model
        mvar_name (str): mvar name defined in gurobi model
        dim (list): dimension of mvar. For 1D mvar, dim = [i]. For 2D mvar, dim = [i,j]

    Returns:
        dict: a dictionary of mvar, which links the original mvar name to a new name that can be used in external functions
    """
    mvars_ = {}
    if len(shape) == 1:
        for i in range(shape[0]):
            mvars_[i] = model.getVarByName(mvar_name + "[%d]" % (i))    
                 
    elif len(shape) == 2:
        for i in range(shape[0]):
            for j in range(shape[1]):
                mvars_[i,j] = model.getVarByName(mvar_name + "[%d,%d]" % (i,j))  
                      
    else:
        raise ValueError("Currently only 1D and 2D mvars are supported")
        
    return mvars_


def _getX_MvarByName(model:gp.Model,mvar_name:str,shape:list) -> np.ndarray:
    
    dic = _getMvarByName(model, mvar_name, shape)
    res_X = np.zeros(shape)
    if len(shape) == 1:
        for i, value in dic.items():
            res_X[i] = value.X
    
    elif len(shape) == 2:
        for (i,j), value in dic.items():
            res_X[i,j] = value.X
    
    else:
        raise ValueError("Currently only 1D and 2D mvars are supported")
    
    return res_X


def _fixMvar(model:gp.Model, mvar_name:str, shape:list, value:np.ndarray, cons_name:str) -> None:
    
    dict_mvar = _getMvarByName(model, mvar_name, shape)
    if value.ndim == 1:
        if model.getConstrByName(cons_name + "[0]") is not None:
            _removeMvarConstrs(model, cons_name, shape)
            
        model.addConstrs((dict_mvar[i] == value[i] for i in range(len(value))), name=cons_name)
        
    elif value.ndim == 2:
        if model.getConstrByName(cons_name + "[0,0]") is not None:
            _removeMvarConstrs(model, cons_name, shape)
            
        model.addConstrs((dict_mvar[i,j] == value[i,j] for i in range(value.shape[0]) for j in range(value.shape[1])), name=cons_name)
        
    else:
        raise ValueError("Currently only 1D and 2D mvars are supported")
    
    model.update()
    

def _removeMvarConstrs(model:gp.Model, cons_name:str, shape:list) -> None:
    
    cons = {}
    if len(shape) == 1:
        for i in range(shape[0]):
            cons[i] = model.getConstrByName(cons_name + "[%d]" %(i))
            
    elif len(shape) == 2:
        for i in range(shape[0]):
            for j in range(shape[1]):
                cons[i, j] = model.getConstrByName(cons_name + "[%d,%d]" %(i, j))
    else:
        raise ValueError("Currently only 1D and 2D mvars are supported")
                
    model.remove(cons)
    model.update()