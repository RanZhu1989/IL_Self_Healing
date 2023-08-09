# 专家轨迹生成，求解NT时间间隔的决策
# 需要外部输入     a: 普通线路健康状态；  X_tieline0：联络线起始状态

using JuMP, Gurobi



expert_model = Model(Gurobi.Optimizer)

@variable(expert_model, PF[1:N_Branch, 1:NT]) # active power flow at line ij
@variable(expert_model, QF[1:N_Branch, 1:NT]) # reactive power flow at line ij
@variable(expert_model, V[1:N_Bus, 1:NT]) # voltage at bus j

@variable(expert_model, P_dg[1:N_DG, 1:NT]) # 上游
@variable(expert_model, Q_dg[1:N_DG, 1:NT])
@variable(expert_model, delta_Qdg[1:N_DG-1, 1:NT-1]) # 去掉上游

@variable(expert_model, Pd_rec[1:N_Bus, 1:NT])
@variable(expert_model, Qd_rec[1:N_Bus, 1:NT])
@variable(expert_model, FF[1:N_Branch, 1:NT]) # commodity flow at line ij

@variable(expert_model, X_rec[1:N_Bus, 1:NT], Bin) # load pick up
@variable(expert_model, X_EN[1:N_Bus, 1:NT], Bin)
@variable(expert_model, X_tieline[1:N_TL, 1:NT], Bin) # line final state. 对于非TieLine,取决于是否受灾，为常数；对于TieLine则作为变量来控制开关
@variable(expert_model, X_line[1:N_NL, 1:NT], Bin)

@variable(expert_model, z_bs[1:N_Bus, 1:NT], Bin) # 用于MP中两个整数变量X相乘的情况
@variable(expert_model, b[1:N_Branch, 1:NT], Bin) # switch state of line ij
@variable(expert_model, X_BS[1:N_Bus, 1:NT], Bin) # 节点是否获得黑启动能力
@variable(expert_model, z_bs1[1:N_Bus, 1:NT], Bin) # 节点是否黑启动条件判断
@variable(expert_model, z_bs2[1:N_Bus, 1:NT], Bin)
@variable(expert_model, z_dg[1:N_DG-1, 1:NT-1], Bin) # 稳定SC输出

# ------------------潮流--------------------
# 1. Bus PQ Blance: S_jk - S_ij = S_inj
@constraint(expert_model, pIn * PF .== DG_Mask * P_dg .- Pd_rec)  # 添加约束
@constraint(expert_model, pIn * QF .== DG_Mask * Q_dg .- Qd_rec)

# 2. Voltage : U_j - U_i = r*Q_ij + x*P_ij
@constraint(expert_model, pIn' * V .- R_Branch .* PF .- X_Branch .* QF .<= Big_M_V .* (ones(size(b)) .- b))
@constraint(expert_model, pIn' * V .- R_Branch .* PF .- X_Branch .* QF .>= -Big_M_V .* (ones(size(b)) .- b))
@constraint(expert_model, X_BS .+ X_EN .* V_min .- z_bs .* V_min .<= V)
@constraint(expert_model, V .<= X_BS .+ X_EN .* V_max .- z_bs .* V_max)
@constraint(expert_model, z_bs .<= X_BS)
@constraint(expert_model, z_bs .<= X_EN)
@constraint(expert_model, z_bs .>= X_BS .+ X_EN .- ones(size(X_BS)))

# 3. % 3. Load Curtailments
@constraint(expert_model, X_rec .<= X_EN)
@constraint(expert_model, X_rec[1,:] .== 0)
@constraint(expert_model, Pd_rec .== X_rec .* Pd)
@constraint(expert_model, Qd_rec .== X_rec .* Qd)
@constraint(expert_model, X_rec[:,2:NT] .>= X_rec[:,1:NT-1])

# % 4. 线路
@constraint(expert_model, PF .>= -S_Branch .* b)
@constraint(expert_model, PF .<= S_Branch .* b)
@constraint(expert_model, QF .>= -S_Branch .* b)
@constraint(expert_model, QF .<= S_Branch .* b)

# ------------DG ----------------
@constraint(expert_model, P_dg .>= P_DG_min) # TODO 后面加上乘以带电状态
@constraint(expert_model, P_dg .<= P_DG_max)
@constraint(expert_model, Q_dg .>= Q_DG_min)
@constraint(expert_model, Q_dg .<= Q_DG_max)

# 由于是固定时间断面，针对SVC可能存在多解
@constraint(expert_model, BigM_SC .* (ones(size(z_dg)) .- z_dg) .<= Q_dg[2:N_DG,2:NT] .- Q_dg[2:N_DG,1:NT-1])
@constraint(expert_model, -BigM_SC .* (ones(size(z_dg)) .- z_dg) .<= delta_Qdg .- (Q_dg[2:N_DG,2:NT] .- Q_dg[2:N_DG,1:NT-1]))
@constraint(expert_model, BigM_SC .* (ones(size(z_dg)) .- z_dg) .>= delta_Qdg .- (Q_dg[2:N_DG,2:NT] .- Q_dg[2:N_DG,1:NT-1]))
@constraint(expert_model, -BigM_SC .* z_dg .<= delta_Qdg .+ (Q_dg[2:N_DG,2:NT] .- Q_dg[2:N_DG,1:NT-1]))
@constraint(expert_model, BigM_SC .* z_dg .>= delta_Qdg .+ (Q_dg[2:N_DG,2:NT] .- Q_dg[2:N_DG,1:NT-1]))

# ---------------Island----------------
#  1. 一个节点为黑启动节点的条件：存在一个BSDG 
@constraint(expert_model, X_BS .<= repeat(sum(BSDG_Mask,dims=2),1,NT))

# % 2. 每个孤岛是联通的。根据节点是否为黑启动节点，分为两种情况讨论
@constraint(expert_model, pIn * FF .+ X_EN .<= Big_M_FF .* (ones(size(z_bs1)) .- z_bs1))
@constraint(expert_model, pIn * FF .+ X_EN .>= -Big_M_FF .* (ones(size(z_bs1)) .- z_bs1))
@constraint(expert_model, z_bs1 .- ones(size(z_bs1)) .<= X_BS)
@constraint(expert_model, X_BS .<= ones(size(z_bs1)) .- z_bs1)
@constraint(expert_model, pIn * FF .>= -Big_M_FF .* (ones(size(z_bs2)) .- z_bs2))
@constraint(expert_model, z_bs2 .- ones(size(z_bs2)) .<= X_BS .- ones(size(X_BS)))
@constraint(expert_model, X_BS .- ones(size(X_BS)) .<= ones(size(z_bs2)) .- z_bs2)
@constraint(expert_model, X_EN .- X_BS .>= -Big_M_FF .* (ones(size(z_bs2)) .- z_bs2))
@constraint(expert_model, X_EN .- X_BS .<= Big_M_FF .* (ones(size(z_bs2)) .- z_bs2))
@constraint(expert_model, z_bs1 .+ z_bs2 .== ones(size(z_bs2)) )

# % 3. 商品流与线路状态
@constraint(expert_model, -Big_M_FF .* b .<= FF)
@constraint(expert_model, FF .<= Big_M_FF .* b)
@constraint(expert_model, b .== [X_line; X_tieline]) # b=[Xline; Xtieline]为全体线路状态，X_line是变量，由a决定，a是外部输入的普通线路健康状态
@constraint(expert_model, X_line .<= a)

#  4. 闭合的边数=总节点数-带电孤岛数-不带电孤立节点数
@constraint(expert_model, sum(b, dims=1) .== N_Bus .- sum(X_BS, dims=1) .- sum(ones(size(X_EN)) .- X_EN, dims=1))

# % 线路操作约束
@constraint(expert_model, X_tieline[:, 2:NT] .>= X_tieline[:, 1:NT-1])
@constraint(expert_model, X_tieline[:, 1] .>= X_tieline0)
@constraint(expert_model, sum(X_tieline[:, 2:NT] .- X_tieline[:, 1:NT-1], dims=1) .<= ones(1,NT-1))
@constraint(expert_model, sum(X_tieline[:, 1] .- X_tieline0, dims=1) .<= 1)

# Obj
@objective(expert_model, Min, -sum(Pd_rec[:]) - 0.01*sum(X_line[:]) + 0.01*sum(delta_Qdg[:]))
optimize!(expert_model)

# RES
objective_value(expert_model)

res_v = value.(V)
res_b = value.(b)
res_Xen = value.(X_EN)
res_Xtie = value.(X_tieline)
res_Qdg = value.(Q_dg)
res_total_Prec = value.(sum(Pd_rec[:]))

# using Plots
# plot(res_v, marker=:circle, linestyle=:solid)


