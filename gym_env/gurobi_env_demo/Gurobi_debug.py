
from System_Data import *
import numpy as np
import gurobipy as gp
from gurobipy import GRB

from julia import Main as jl


system_data = System_Data(file_name='Case_33BW_Data.xlsx')
NT, N_Branch, N_TL, N_NL, N_Bus, pIn, N_DG, DG_Mask, R_Branch, X_Branch, Big_M_V, V0, \
    V_min, V_max, Pd, Qd, S_Branch, P_DG_min, P_DG_max, Q_DG_min, Q_DG_max, BigM_SC, BSDG_Mask, \
    Big_M_FF = system_data.args_expert

"""The following code is to show that the Gurobi model is not working properly. On the other hand, the JuMP model is working properly.
"""

# Gurobi Model
model = gp.Model("Expert_Model")

PF = model.addMVar(shape=(N_Branch,NT), vtype=GRB.CONTINUOUS, name="PF")
QF = model.addMVar(shape=(N_Branch,NT), vtype=GRB.CONTINUOUS, name="QF")

P_dg = model.addMVar(shape=(N_DG,NT), vtype=GRB.CONTINUOUS, name="P_dg")
Q_dg = model.addMVar(shape=(N_DG,NT), vtype=GRB.CONTINUOUS, name="Q_dg")

b = model.addMVar(shape=(N_Branch,NT), vtype=GRB.BINARY, name="b")


model.addConstr(pIn @ PF - DG_Mask @ P_dg == -Pd)
model.addConstr(pIn @ QF - DG_Mask @ Q_dg == -Qd)

model.addConstr(PF + S_Branch * b >= 0)
model.addConstr(PF - S_Branch * b <= 0)
model.addConstr(QF + S_Branch * b >= 0)
model.addConstr(QF - S_Branch * b <= 0)

model.addConstr(P_dg >= P_DG_min)
model.addConstr(P_dg <= P_DG_max)
model.addConstr(Q_dg >= Q_DG_min)
model.addConstr(Q_dg <= Q_DG_max)
model.addConstr(Q_dg[0,:] == 0.019)
model.addConstr(Q_dg[1,:] == 0.002)
model.addConstr(Q_dg[2,:] == -0.002)
model.addConstr(Q_dg[3,:] == -0.002)
model.addConstr(Q_dg[3:,:] == 0.002)

model.addConstr(b[N_NL:,:] == 0)
model.addConstr(b[0:N_NL,:] == 1)

model.setObjective(0)

model.optimize()

# JuMP Model
jl.include("demo_julia.jl")
jl.test(system_data.args_expert)

