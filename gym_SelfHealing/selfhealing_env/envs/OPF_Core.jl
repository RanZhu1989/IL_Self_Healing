using JuMP

struct OPF_Core
    expert_model
    step_model
    reset_model
    # 在一次任务中，OPF_Core只需初始化一次，关于系统的data在创建时输入，env.rest会产生a, 
    function OPF_Core(args_expert, args_step)
        expert_model = make_expert_model(args_expert)
        step_model = make_step_model(args_step)
        reset_model = make_reset_model(args_step)
        new(expert_model, step_model, reset_model)
    end

end


function make_expert_model(args_expert)

    NT, N_Branch, N_TL, N_NL, N_Bus, pIn, N_DG, DG_Mask, R_Branch, X_Branch, Big_M_V, V0,
        V_min, V_max, Pd, Qd, S_Branch, P_DG_min, P_DG_max, Q_DG_min, Q_DG_max, BigM_SC, BSDG_Mask,
        Big_M_FF = args_expert
    
    expert_model = Model()

    # --- 需要经常改变的常量 ---
    # 找了一下，JuMP对约束的修改不支持张量，觉得还是fix比较适合
    @variable(expert_model, a[1:N_NL, 1:NT]) # line health state
    @variable(expert_model, X_tieline0[1:N_TL]) 


    # --------------------

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
    @constraint(expert_model, V0 .* (pIn' * V) .- R_Branch .* PF .- X_Branch .* QF .<= Big_M_V .* (1 .- b))
    @constraint(expert_model, V0 .* (pIn' * V) .- R_Branch .* PF .- X_Branch .* QF .>= -Big_M_V .* (1 .- b))
    @constraint(expert_model, X_BS .+ X_EN .* V_min .- z_bs .* V_min .<= V)
    @constraint(expert_model, V .<= X_BS .* V0 .+ X_EN .* V_max .- z_bs .* V_max)
    @constraint(expert_model, z_bs .<= X_BS)
    @constraint(expert_model, z_bs .<= X_EN)
    @constraint(expert_model, z_bs .>= X_BS .+ X_EN .- 1)

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
    @constraint(expert_model, P_dg .>= (DG_Mask'*X_EN) .* P_DG_min) 
    @constraint(expert_model, P_dg .<= (DG_Mask'*X_EN) .* P_DG_max)
    @constraint(expert_model, Q_dg .>= (DG_Mask'*X_EN) .* Q_DG_min)
    @constraint(expert_model, Q_dg .<= (DG_Mask'*X_EN) .* Q_DG_max)

    # 由于是固定时间断面，针对SVC可能存在多解
    @constraint(expert_model, BigM_SC .* (1 .- z_dg) .<= Q_dg[2:N_DG,2:NT] .- Q_dg[2:N_DG,1:NT-1])
    @constraint(expert_model, -BigM_SC .* (1 .- z_dg) .<= delta_Qdg .- (Q_dg[2:N_DG,2:NT] .- Q_dg[2:N_DG,1:NT-1]))
    @constraint(expert_model, BigM_SC .* (1 .- z_dg) .>= delta_Qdg .- (Q_dg[2:N_DG,2:NT] .- Q_dg[2:N_DG,1:NT-1]))
    @constraint(expert_model, -BigM_SC .* z_dg .<= delta_Qdg .+ (Q_dg[2:N_DG,2:NT] .- Q_dg[2:N_DG,1:NT-1]))
    @constraint(expert_model, BigM_SC .* z_dg .>= delta_Qdg .+ (Q_dg[2:N_DG,2:NT] .- Q_dg[2:N_DG,1:NT-1]))

    # ---------------Island----------------
    #  1. 一个节点为黑启动节点的条件：存在一个BSDG 
    @constraint(expert_model, X_BS .<= repeat(sum(BSDG_Mask,dims=2),1,NT))

    # % 2. 每个孤岛是联通的。根据节点是否为黑启动节点，分为两种情况讨论
    @constraint(expert_model, pIn * FF .+ X_EN .<= Big_M_FF .* (1 .- z_bs1))
    @constraint(expert_model, pIn * FF .+ X_EN .>= -Big_M_FF .* (1 .- z_bs1))
    @constraint(expert_model, z_bs1 .- 1 .<= X_BS)
    @constraint(expert_model, X_BS .<= 1 .- z_bs1)
    @constraint(expert_model, pIn * FF .>= -Big_M_FF .* (1 .- z_bs2))
    @constraint(expert_model, z_bs2 .- 1 .<= X_BS .- 1)
    @constraint(expert_model, X_BS .- 1 .<= 1 .- z_bs2)
    @constraint(expert_model, X_EN .- X_BS .>= -Big_M_FF .* (1 .- z_bs2))
    @constraint(expert_model, X_EN .- X_BS .<= Big_M_FF .* (1 .- z_bs2))
    @constraint(expert_model, z_bs1 .+ z_bs2 .== 1 )

    # % 3. 商品流与线路状态
    @constraint(expert_model, -Big_M_FF .* b .<= FF)
    @constraint(expert_model, FF .<= Big_M_FF .* b)
    @constraint(expert_model, b .== [X_line; X_tieline]) # b=[Xline; Xtieline]为全体线路状态，X_line是变量，由a决定，a是外部输入的普通线路健康状态
    @constraint(expert_model, X_line .<= a ) #NOTE a 需要在外部输入 这个仅在env.rest才需要

    #  4. 闭合的边数=总节点数-带电孤岛数-不带电孤立节点数
    @constraint(expert_model, sum(b, dims=1) .== N_Bus .- sum(X_BS, dims=1) .- sum(1 .- X_EN, dims=1))

    # % 线路操作约束
    @constraint(expert_model, X_tieline[:, 2:NT] .>= X_tieline[:, 1:NT-1])

    @constraint(expert_model, X_tieline[:, 1] .>= X_tieline0) #NOTE X_tieline0 需要在外部输入
    @constraint(expert_model, sum(X_tieline[:, 2:NT] .- X_tieline[:, 1:NT-1], dims=1) .<= 1)
    @constraint(expert_model, sum(X_tieline[:, 1] .- X_tieline0, dims=1) .<= 1) #NOTE X_tieline0 需要在外部输入

    # Obj
    @objective(expert_model, Min, -sum(Pd_rec[:]) - 0.01*sum(X_line[:]) + sum(delta_Qdg[:]))

    return expert_model
    
end


function make_step_model(args_step)
    NT, N_Branch, N_TL, N_NL, N_Bus, pIn, N_DG, DG_Mask, R_Branch, X_Branch, Big_M_V, V0,
        V_min, V_max, Pd, Qd, S_Branch, P_DG_min, P_DG_max, Q_DG_min,
        Q_DG_max, BSDG_Mask, Big_M_FF = args_step


    step_model = Model()

    # --- 需要经常改变的常量 ---
    # 找了一下，JuMP对约束的修改不支持张量，觉得还是fix比较适合
    @variable(step_model, a[1:N_NL]) # line health state
    @variable(step_model, X_rec0[1:N_Bus]) 
    @variable(step_model, Q_svc[1:N_DG-1])


    
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
    @constraint(step_model, V0 .* (pIn' * V) .- R_Branch .* PF .- X_Branch .* QF .<= Big_M_V .* (1 .- b))
    @constraint(step_model, V0 .* (pIn' * V) .- R_Branch .* PF .- X_Branch .* QF .>= -Big_M_V .* (1 .- b))
    @constraint(step_model, X_BS .+ X_EN .* V_min .- z_bs .* V_min .<= V)
    @constraint(step_model, V .<= X_BS .* V0 .+ X_EN .* V_max .- z_bs .* V_max)
    @constraint(step_model, z_bs .<= X_BS)
    @constraint(step_model, z_bs .<= X_EN)
    @constraint(step_model, z_bs .>= X_BS .+ X_EN .- 1)

    # 3. % 3. Load Curtailments
    @constraint(step_model, X_rec .<= X_EN)
    @constraint(step_model, X_rec[1,:] .== 0)
    @constraint(step_model, Pd_rec .== X_rec .* Pd)
    @constraint(step_model, Qd_rec .== X_rec .* Qd)
    @constraint(step_model, X_rec[:,1] .>= X_rec0) # step版本 恢复的不能失去  X_rec0需要从外部输入 #NOTE X_rec0 需要在外部输入

    # % 4. 线路
    @constraint(step_model, PF .>= -S_Branch .* b)
    @constraint(step_model, PF .<= S_Branch .* b)
    @constraint(step_model, QF .>= -S_Branch .* b)
    @constraint(step_model, QF .<= S_Branch .* b)

    # ------------DG ----------------
    @constraint(step_model, P_dg .>= (DG_Mask'*X_EN) .* P_DG_min) # 这里要再调度一下
    @constraint(step_model, P_dg .<= (DG_Mask'*X_EN) .* P_DG_max)
    @constraint(step_model, Q_dg .>= (DG_Mask'*X_EN) .* Q_DG_min)
    @constraint(step_model, Q_dg .<= (DG_Mask'*X_EN) .* Q_DG_max)
    
    @constraint(step_model, e_Qsvc .== Q_svc .- Q_dg[2:N_DG,1] ) # 取除了第一个DG外的无功Q_dg与指令比较 #NOTE Q_svc_input 需要在外部输入
    @constraint(step_model, e_Qsvc .== e_Qsvc_up .- e_Qsvc_down) # 用于绝对值
    @constraint(step_model, e_Qsvc_up .>= 0)
    @constraint(step_model, e_Qsvc_down .>= 0)
    # @constraint(step_model, e_Qsvc .== 0) #NOTE 测试严格等于用

    # ---------------Island----------------
    #  1. 一个节点为黑启动节点的条件：存在一个BSDG 
    @constraint(step_model, X_BS .<= repeat(sum(BSDG_Mask,dims=2),1,NT))

    # % 2. 每个孤岛是联通的。根据节点是否为黑启动节点，分为两种情况讨论
    @constraint(step_model, pIn * FF .+ X_EN .<= Big_M_FF .* (1 .- z_bs1))
    @constraint(step_model, pIn * FF .+ X_EN .>= -Big_M_FF .* (1 .- z_bs1))
    @constraint(step_model, z_bs1 .- 1 .<= X_BS)
    @constraint(step_model, X_BS .<= 1 .- z_bs1)
    @constraint(step_model, pIn * FF .>= -Big_M_FF .* (1 .- z_bs2))
    @constraint(step_model, z_bs2 .- 1 .<= X_BS .- 1)
    @constraint(step_model, X_BS .- 1 .<= 1 .- z_bs2)
    @constraint(step_model, X_EN .- X_BS .>= -Big_M_FF .* (1 .- z_bs2))
    @constraint(step_model, X_EN .- X_BS .<= Big_M_FF .* (1 .- z_bs2))
    @constraint(step_model, z_bs1 .+ z_bs2 .== 1 )

    # % 3. 商品流与线路状态
    @constraint(step_model, -Big_M_FF .* b .<= FF)
    @constraint(step_model, FF .<= Big_M_FF .* b)
    @constraint(step_model, b .== [X_line; X_tieline]) # % b=[Xline; Xtieline]为全体线路状态，X_line是变量，由a决定，a是外部输入的普通线路健康状态
    @constraint(step_model, X_line .<= a) #NOTE a 需要在外部输入  这个仅在env.rest才需要

    #  4. 闭合的边数=总节点数-带电孤岛数-不带电孤立节点数
    @constraint(step_model, sum(b, dims=1) .== N_Bus .- sum(X_BS, dims=1) .- sum(1 .- X_EN, dims=1))

    
    # Obj
    @objective(step_model, Min, -sum(Pd_rec[:]) - 0.01*sum(X_line[:]) + 10*sum(e_Qsvc_up .+ e_Qsvc_down))

    return step_model
end


function make_reset_model(args_step)
    NT, N_Branch, N_TL, N_NL, N_Bus, pIn, N_DG, DG_Mask, R_Branch, X_Branch, Big_M_V, V0,
        V_min, V_max, Pd, Qd, S_Branch, P_DG_min, P_DG_max, Q_DG_min,
        Q_DG_max, BSDG_Mask, Big_M_FF = args_step

    reset_model = Model()
    # --- 需要经常改变的常量 ---
    # 找了一下，JuMP对约束的修改不支持张量，觉得还是fix比较适合
    @variable(reset_model, a[1:N_NL]) # line health state
    @variable(reset_model, Q_svc[1:N_DG-1])


    @variable(reset_model, PF[1:N_Branch, 1:NT]) # active power flow at line ij
    @variable(reset_model, QF[1:N_Branch, 1:NT]) # reactive power flow at line ij
    @variable(reset_model, V[1:N_Bus, 1:NT]) # voltage at bus j

    @variable(reset_model, P_dg[1:N_DG, 1:NT]) # 上游
    @variable(reset_model, Q_dg[1:N_DG, 1:NT])

    @variable(reset_model, Pd_rec[1:N_Bus, 1:NT])
    @variable(reset_model, Qd_rec[1:N_Bus, 1:NT])
    @variable(reset_model, FF[1:N_Branch, 1:NT]) # commodity flow at line ij

    @variable(reset_model, X_rec[1:N_Bus, 1:NT], Bin) # load pick up
    @variable(reset_model, X_EN[1:N_Bus, 1:NT], Bin)
    @variable(reset_model, X_tieline[1:N_TL, 1:NT], Bin) #NOTE 这个需要输入全不动的初始状态 和tieline0其实是一样的
    @variable(reset_model, X_line[1:N_NL, 1:NT], Bin)

    @variable(reset_model, z_bs[1:N_Bus, 1:NT], Bin) # 用于MP中两个整数变量X相乘的情况
    @variable(reset_model, b[1:N_Branch, 1:NT], Bin) # switch state of line ij
    @variable(reset_model, X_BS[1:N_Bus, 1:NT], Bin) # 节点是否获得黑启动能力
    @variable(reset_model, z_bs1[1:N_Bus, 1:NT], Bin) # 节点是否黑启动条件判断
    @variable(reset_model, z_bs2[1:N_Bus, 1:NT], Bin)


        # ------------------潮流--------------------
    # 1. Bus PQ Blance: S_jk - S_ij = S_inj
    @constraint(reset_model, pIn * PF .== DG_Mask * P_dg .- Pd_rec)  # 添加约束
    @constraint(reset_model, pIn * QF .== DG_Mask * Q_dg .- Qd_rec)

    # 2. Voltage : U_j - U_i = r*Q_ij + x*P_ij
    @constraint(reset_model, V0 .* (pIn' * V) .- R_Branch .* PF .- X_Branch .* QF .<= Big_M_V .* (1 .- b))
    @constraint(reset_model, V0 .* (pIn' * V) .- R_Branch .* PF .- X_Branch .* QF .>= -Big_M_V .* (1 .- b))
    @constraint(reset_model, X_BS .+ X_EN .* V_min .- z_bs .* V_min .<= V)
    @constraint(reset_model, V .<= X_BS .* V0 .+ X_EN .* V_max .- z_bs .* V_max)
    @constraint(reset_model, z_bs .<= X_BS)
    @constraint(reset_model, z_bs .<= X_EN)
    @constraint(reset_model, z_bs .>= X_BS .+ X_EN .- 1)

    # 3. % 3. Load Curtailments
    @constraint(reset_model, X_rec .<= X_EN)
    @constraint(reset_model, X_rec[1,:] .== 0)
    @constraint(reset_model, Pd_rec .== X_rec .* Pd)
    @constraint(reset_model, Qd_rec .== X_rec .* Qd)

    # % 4. 线路
    @constraint(reset_model, PF .>= -S_Branch .* b)
    @constraint(reset_model, PF .<= S_Branch .* b)
    @constraint(reset_model, QF .>= -S_Branch .* b)
    @constraint(reset_model, QF .<= S_Branch .* b)

    # ------------DG ----------------
    @constraint(reset_model, P_dg .>= (DG_Mask'*X_EN) .* P_DG_min) # 这里要再调度一下
    @constraint(reset_model, P_dg .<= (DG_Mask'*X_EN) .* P_DG_max)
    @constraint(reset_model, Q_dg .>= (DG_Mask'*X_EN) .* Q_DG_min)
    @constraint(reset_model, Q_dg .<= (DG_Mask'*X_EN) .* Q_DG_max)
    
    @constraint(reset_model, Q_dg[2:N_DG,1] .== Q_svc) #  #NOTE Q_svc_input 需要在外部输入

    # ---------------Island----------------
    #  1. 一个节点为黑启动节点的条件：存在一个BSDG 
    @constraint(reset_model, X_BS .<= repeat(sum(BSDG_Mask,dims=2),1,NT))

    # % 2. 每个孤岛是联通的。根据节点是否为黑启动节点，分为两种情况讨论
    @constraint(reset_model, pIn * FF .+ X_EN .<= Big_M_FF .* (1 .- z_bs1))
    @constraint(reset_model, pIn * FF .+ X_EN .>= -Big_M_FF .* (1 .- z_bs1))
    @constraint(reset_model, z_bs1 .- 1 .<= X_BS)
    @constraint(reset_model, X_BS .<= 1 .- z_bs1)
    @constraint(reset_model, pIn * FF .>= -Big_M_FF .* (1 .- z_bs2))
    @constraint(reset_model, z_bs2 .- 1 .<= X_BS .- 1)
    @constraint(reset_model, X_BS .- 1 .<= 1 .- z_bs2)
    @constraint(reset_model, X_EN .- X_BS .>= -Big_M_FF .* (1 .- z_bs2))
    @constraint(reset_model, X_EN .- X_BS .<= Big_M_FF .* (1 .- z_bs2))
    @constraint(reset_model, z_bs1 .+ z_bs2 .== 1 )

    # % 3. 商品流与线路状态
    @constraint(reset_model, -Big_M_FF .* b .<= FF)
    @constraint(reset_model, FF .<= Big_M_FF .* b)
    @constraint(reset_model, b .== [X_line; X_tieline]) # % b=[Xline; Xtieline]为全体线路状态，X_line是变量，由a决定，a是外部输入的普通线路健康状态
    @constraint(reset_model, X_line .<= a) #NOTE a 需要在外部输入  这个仅在env.rest才需要

    #  4. 闭合的边数=总节点数-带电孤岛数-不带电孤立节点数
    @constraint(reset_model, sum(b, dims=1) .== N_Bus .- sum(X_BS, dims=1) .- sum(1 .- X_EN, dims=1))

    
    # Obj
    @objective(reset_model, Min, -sum(Pd_rec[:]) - 0.01*sum(X_line[:]))

    return reset_model

end


function init_opf_core(; args_expert, args_step, solver="CPLEX", display=true)
    """We can not use jl.OPF_Core() thourgh julia_python interface. 
    Therefore we use this function to initialize the core."""

    global core = OPF_Core(args_expert, args_step)
    # NOTE MIP精度也在这里设置
    if solver == "CPLEX"
        optimizer = optimizer_with_attributes(CPLEX.Optimizer,"CPXPARAM_ScreenOutput" => display)
    elseif solver == "Gurobi"
        optimizer = optimizer_with_attributes(Gurobi.Optimizer,"output_flag" => display)
    else
        error("solver not supported")
    end

    set_optimizer(core.expert_model, optimizer)
    set_optimizer(core.step_model, optimizer)
    set_optimizer(core.reset_model, optimizer)

end


function set_dmg(a_input)
    # 重设线路故障状态
    global core

    for idx in eachindex(a_input)
        fix(core.expert_model[:a][idx], a_input[idx]) # 通过fix的方法更加灵活
    end

    for idx in eachindex(a_input[:,1])
        fix(core.step_model[:a][idx], a_input[idx,1])
        fix(core.reset_model[:a][idx], a_input[idx,1])
    end

end

function set_ExpertModel(; X_tieline0_input, vvo=true)
    # 设置tieline初始状态
    global core

    for idx in eachindex(X_tieline0_input)
        fix(core.expert_model[:X_tieline0][idx],X_tieline0_input[idx])
    end

    if !vvo
        for idx in eachindex(core.expert_model[:Q_dg][2:end])
            fix(core.expert_model[:Q_dg][2:end][idx],0)
        end
    end
end

function set_StepModel(; X_rec0_input,X_tieline_input,Q_svc_input=nothing, vvo=true)
    # 为step模型输入上一步负荷状态、tieline指令、svc指令

    if vvo && Q_svc_input === nothing
        error("Please provide a value for Q_svc_input when vvo mode is set to true.")
    end

    global core
    
    for idx in eachindex(X_rec0_input)
        fix(core.step_model[:X_rec0][idx],X_rec0_input[idx])
    end

    for idx in eachindex(X_tieline_input)
        fix(core.step_model[:X_tieline][idx],X_tieline_input[idx])
    end

    for idx in eachindex(core.step_model[:Q_svc])
        # 是否考虑svc的区别就在于是否接受输入
        if vvo
            fix(core.step_model[:Q_svc][idx],Q_svc_input[idx])
        else
            fix(core.step_model[:Q_svc][idx],0)
        end
    end
    
end


function set_ResetModel(; X_tieline_input, Q_svc_input)

    global core

    for idx in eachindex(X_tieline_input)
        fix(core.reset_model[:X_tieline][idx],X_tieline_input[idx])
    end

    for idx in eachindex(Q_svc_input)
        fix(core.reset_model[:Q_svc][idx],Q_svc_input[idx])
    end
    
end


function solve_ExpertModel()

    global core
    optimize!(core.expert_model)
    b = value.(core.step_model[:b])

    return objective_value(core.expert_model)

end

function solve_StepModel()
    global core

    solved_flag = false
    
    optimize!(core.step_model)

    if termination_status(core.step_model) == MOI.OPTIMAL
        solved_flag = true
        b = value.(core.step_model[:b][:,1])
        x_tieline = value.(core.step_model[:X_tieline][:,1])
        x_load = value.(core.step_model[:X_rec][:,1])
        PF = value.(core.step_model[:PF][:,1])
        QF = value.(core.step_model[:QF][:,1])
        Prec = sum(value.(core.step_model[:Pd_rec]))
        e_Qvsc = value.(sum(core.step_model[:e_Qsvc_up] .+ core.step_model[:e_Qsvc_down]))
        
        return solved_flag, b, x_tieline, x_load, PF, QF, Prec, e_Qvsc
        
    else
        return solved_flag, Nothing, Nothing, Nothing, Nothing, Nothing, Nothing, Nothing 
    end
end


function solve_ResetModel()
    global Core

    optimize!(core.reset_model)

    # 返回所有线路状态、所有负荷状态、所有线路PQ潮流
    b = value.(core.reset_model[:b][:,1])
    x_tieline = value.(core.reset_model[:X_tieline][:,1])
    x_load = value.(core.reset_model[:X_rec][:,1])
    PF = value.(core.reset_model[:PF][:,1])
    QF = value.(core.reset_model[:QF][:,1])
    Prec = sum(value.(core.reset_model[:Pd_rec]))

    return b, x_tieline, x_load, PF, QF, Prec
end