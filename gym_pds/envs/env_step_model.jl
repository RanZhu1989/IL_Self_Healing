# % 求解单步 NT = 1
# % 需要输入 X_rec0: 上一步负荷pick up情况；X_tieline_input: Tieline控制； Q_svc_input： svc控制

using JuMP, Gurobi


X_rec0 = zeros(33) # 这些是env.step上一步的负荷恢复结果

X_tieline_input = [1; 1; 1; 0; 0] # 这些是env.step输入的tieline动作

Q_svc_input = [0.002; 0.002; 0.000; 0.002; 0.002; 0.000] # 这些是env.step输入的svc动作



step_model = Model(Gurobi.Optimizer)

@variable(step_model, PF[1:N_Branch, 1:NT]) # active power flow at line ij
@variable(step_model, QF[1:N_Branch, 1:NT]) # reactive power flow at line ij
@variable(step_model, V[1:N_Bus, 1:NT]) # voltage at bus j

@variable(step_model, P_dg[1:N_DG, 1:NT]) # 上游
@variable(step_model, Q_dg[1:N_DG, 1:NT])
@variable(step_model, e_Qsvc[1:N_DG-1]) # step需要指令跟踪误差
@variable(step_model, e_Qsvc_up[1:N_DG-1]) # 用于绝对值
@variable(step_model, e_Qsvc_down[1:N_DG-1]) 

@variable(step_model, Pd_rec[1:N_Bus, 1:NT])
@variable(step_model, Qd_rec[1:N_Bus, 1:NT])
@variable(step_model, FF[1:N_Branch, 1:NT]) # commodity flow at line ij

@variable(step_model, X_rec[1:N_Bus, 1:NT], Bin) # load pick up
@variable(step_model, X_EN[1:N_Bus, 1:NT], Bin)
@variable(step_model, X_tieline[1:N_TL, 1:NT], Bin) # line final state. 对于非TieLine,取决于是否受灾，为常数；对于TieLine则作为变量来控制开关
@variable(step_model, X_line[1:N_NL, 1:NT], Bin)

@variable(step_model, z_bs[1:N_Bus, 1:NT], Bin) # 用于MP中两个整数变量X相乘的情况
@variable(step_model, b[1:N_Branch, 1:NT], Bin) # switch state of line ij
@variable(step_model, X_BS[1:N_Bus, 1:NT], Bin) # 节点是否获得黑启动能力
@variable(step_model, z_bs1[1:N_Bus, 1:NT], Bin) # 节点是否黑启动条件判断
@variable(step_model, z_bs2[1:N_Bus, 1:NT], Bin)


# ------------------潮流--------------------
# 1. Bus PQ Blance: S_jk - S_ij = S_inj
@constraint(step_model, pIn * PF .== DG_Mask * P_dg .- Pd_rec)  # 添加约束
@constraint(step_model, pIn * QF .== DG_Mask * Q_dg .- Qd_rec)

# 2. Voltage : U_j - U_i = r*Q_ij + x*P_ij
@constraint(step_model, pIn' * V .- R_Branch .* PF .- X_Branch .* QF .<= Big_M_V .* (ones(size(b)) .- b))
@constraint(step_model, pIn' * V .- R_Branch .* PF .- X_Branch .* QF .>= -Big_M_V .* (ones(size(b)) .- b))
@constraint(step_model, X_BS .+ X_EN .* V_min .- z_bs .* V_min .<= V)
@constraint(step_model, V .<= X_BS .+ X_EN .* V_max .- z_bs .* V_max)
@constraint(step_model, z_bs .<= X_BS)
@constraint(step_model, z_bs .<= X_EN)
@constraint(step_model, z_bs .>= X_BS .+ X_EN .- ones(size(X_BS)))

# 3. % 3. Load Curtailments
@constraint(step_model, X_rec .<= X_EN)
@constraint(step_model, X_rec[1,:] .== 0)
@constraint(step_model, Pd_rec .== X_rec .* Pd)
@constraint(step_model, Qd_rec .== X_rec .* Qd)
@constraint(step_model, X_rec[:1] .>= X_rec0) # step版本 恢复的不能失去  X_rec0需要从外部输入

# % 4. 线路
@constraint(step_model, PF .>= -S_Branch .* b)
@constraint(step_model, PF .<= S_Branch .* b)
@constraint(step_model, QF .>= -S_Branch .* b)
@constraint(step_model, QF .<= S_Branch .* b)

# ------------DG ----------------
@constraint(step_model, P_dg .>= P_DG_min) # 这里要再调度一下
@constraint(step_model, P_dg .<= P_DG_max)
@constraint(step_model, Q_dg .>= Q_DG_min)
@constraint(step_model, Q_dg .<= Q_DG_max)
@constraint(step_model, e_Qsvc .== Q_svc_input .- Q_dg[2:N_DG,1] ) # 取除了第一个DG外的无功Q_dg与指令比较
@constraint(step_model, e_Qsvc .== e_Qsvc_up .- e_Qsvc_down) # 用于绝对值
@constraint(step_model, e_Qsvc_up .>= 0)
@constraint(step_model, e_Qsvc_down .>= 0)

# ---------------Island----------------
#  1. 一个节点为黑启动节点的条件：存在一个BSDG 
@constraint(step_model, X_BS .<= repeat(sum(BSDG_Mask,dims=2),1,NT))

# % 2. 每个孤岛是联通的。根据节点是否为黑启动节点，分为两种情况讨论
@constraint(step_model, pIn * FF .+ X_EN .<= Big_M_FF .* (ones(size(z_bs1)) .- z_bs1))
@constraint(step_model, pIn * FF .+ X_EN .>= -Big_M_FF .* (ones(size(z_bs1)) .- z_bs1))
@constraint(step_model, z_bs1 .- ones(size(z_bs1)) .<= X_BS)
@constraint(step_model, X_BS .<= ones(size(z_bs1)) .- z_bs1)
@constraint(step_model, pIn * FF .>= -Big_M_FF .* (ones(size(z_bs2)) .- z_bs2))
@constraint(step_model, z_bs2 .- ones(size(z_bs2)) .<= X_BS .- ones(size(X_BS)))
@constraint(step_model, X_BS .- ones(size(X_BS)) .<= ones(size(z_bs2)) .- z_bs2)
@constraint(step_model, X_EN .- X_BS .>= -Big_M_FF .* (ones(size(z_bs2)) .- z_bs2))
@constraint(step_model, X_EN .- X_BS .<= Big_M_FF .* (ones(size(z_bs2)) .- z_bs2))
@constraint(step_model, z_bs1 .+ z_bs2 .== ones(size(z_bs2)) )

# % 3. 商品流与线路状态
@constraint(step_model, -Big_M_FF .* b .<= FF)
@constraint(step_model, FF .<= Big_M_FF .* b)
@constraint(step_model, b .== [X_line; X_tieline]) # % b=[Xline; Xtieline]为全体线路状态，X_line是变量，由a决定，a是外部输入的普通线路健康状态
@constraint(step_model, X_line .<= a)

#  4. 闭合的边数=总节点数-带电孤岛数-不带电孤立节点数
@constraint(step_model, sum(b, dims=1) .== N_Bus .- sum(X_BS, dims=1) .- sum(ones(size(X_EN)) .- X_EN, dims=1))

# % 线路操作约束
@constraint(step_model, X_tieline .== X_tieline_input) # % 不需要下面的约束，直接从外部输入

# Obj
@objective(step_model, Min, -sum(Pd_rec[:]) - 0.01*sum(X_line[:]) + 0.01*sum(e_Qsvc_up .+ e_Qsvc_down))

# RES
optimize!(step_model)
objective_value(step_model)

res_v = value.(V)
res_b = value.(b)
res_Xen = value.(X_EN)
res_Xtie = value.(X_tieline)
res_Qdg = value.(Q_dg)
res_total_Prec = value.(sum(Pd_rec[:]))
