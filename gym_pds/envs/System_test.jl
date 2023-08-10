import XLSX
using JuMP, Gurobi
include(joinpath(@__DIR__,"utils_env.jl"))

struct OPF_Core
    expert_model
    step_model
    # 在一次任务中，OPF_Core只需初始化一次，关于系统的data在创建时输入，env.rest会产生a, 
    function OPF_Core(args_expert, args_step)
        expert = make_expert_model(args_expert)
        step = make_step_model(args_step)
        new(expert, step)
    end

end


function make_expert_model(args_expert)

    NT, N_Branch, N_TL, N_NL, N_Bus, pIn, N_DG, DG_Mask, R_Branch, X_Branch, Big_M_V, V0,
        V_min, V_max, Pd, Qd, S_Branch, P_DG_min, P_DG_max, Q_DG_min, Q_DG_max, BigM_SC, BSDG_Mask,
        Big_M_FF, a_input, X_tieline0_input = args_expert
    
    println(pIn)
    println(N_NL)
    println(NT)
    expert_model = Model()

    # --- 需要经常改变的常量 ---
    # 找了一下，JuMP对约束的修改不支持张量，觉得还是fix比较适合
    @variable(expert_model, a[1:N_NL, 1:NT]) # line health state
    @variable(expert_model, X_tieline0[1:N_TL]) 

    for i in 1:NT
        for j in 1:N_NL
            fix(a[j,i],a_input[j,i])
        end
    end

    for j in 1:N_TL
        fix(X_tieline0[j],X_tieline0_input[j])
    end

    

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
    @constraint(expert_model, pIn' * V .- R_Branch .* PF .- X_Branch .* QF .<= Big_M_V .* (ones(size(b)) .- b))
    @constraint(expert_model, pIn' * V .- R_Branch .* PF .- X_Branch .* QF .>= -Big_M_V .* (ones(size(b)) .- b))
    @constraint(expert_model, X_BS .+ X_EN .* V_min .- z_bs .* V_min .<= V)
    @constraint(expert_model, V .<= X_BS .* V0 .+ X_EN .* V_max .- z_bs .* V_max)
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
    @constraint(expert_model, P_dg .>= (DG_Mask'*X_EN) .* P_DG_min) 
    @constraint(expert_model, P_dg .<= (DG_Mask'*X_EN) .* P_DG_max)
    @constraint(expert_model, Q_dg .>= (DG_Mask'*X_EN) .* Q_DG_min)
    @constraint(expert_model, Q_dg .<= (DG_Mask'*X_EN) .* Q_DG_max)

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
    @constraint(expert_model, X_line .<= a ) #NOTE a 需要在外部输入 这个仅在env.rest才需要

    #  4. 闭合的边数=总节点数-带电孤岛数-不带电孤立节点数
    @constraint(expert_model, sum(b, dims=1) .== N_Bus .- sum(X_BS, dims=1) .- sum(ones(size(X_EN)) .- X_EN, dims=1))

    # % 线路操作约束
    @constraint(expert_model, X_tieline[:, 2:NT] .>= X_tieline[:, 1:NT-1])

    @constraint(expert_model, X_tieline[:, 1] .>= X_tieline0) #NOTE X_tieline0 需要在外部输入
    @constraint(expert_model, sum(X_tieline[:, 2:NT] .- X_tieline[:, 1:NT-1], dims=1) .<= ones(1,NT-1))
    @constraint(expert_model, sum(X_tieline[:, 1] .- X_tieline0, dims=1) .<= 1) #NOTE X_tieline0 需要在外部输入




    # Obj
    @objective(expert_model, Min, -sum(Pd_rec[:]) - 0.01*sum(X_line[:]) + 0.01*sum(delta_Qdg[:]))

    return expert_model
    
end


function make_step_model(args_step)
    NT, N_Branch, N_TL, N_NL, N_Bus, pIn, N_DG, DG_Mask, R_Branch, X_Branch, Big_M_V, V0,
        V_min, V_max, Pd, Qd, S_Branch, P_DG_min, P_DG_max, Q_DG_min,
        Q_DG_max, BSDG_Mask, Big_M_FF, a, X_rec0, X_tieline_input, Q_svc_input = args_step


    step_model = Model()

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
    @constraint(step_model, V .<= X_BS .* V0 .+ X_EN .* V_max .- z_bs .* V_max)
    @constraint(step_model, z_bs .<= X_BS)
    @constraint(step_model, z_bs .<= X_EN)
    @constraint(step_model, z_bs .>= X_BS .+ X_EN .- ones(size(X_BS)))

    # 3. % 3. Load Curtailments
    @constraint(step_model, X_rec .<= X_EN)
    @constraint(step_model, X_rec[1,:] .== 0)
    @constraint(step_model, Pd_rec .== X_rec .* Pd)
    @constraint(step_model, Qd_rec .== X_rec .* Qd)
    @constraint(step_model, X_rec[:1] .>= X_rec0) # step版本 恢复的不能失去  X_rec0需要从外部输入 #NOTE X_tieline0 需要在外部输入

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
    @constraint(step_model, e_Qsvc .== Q_svc_input .- Q_dg[2:N_DG,1] ) # 取除了第一个DG外的无功Q_dg与指令比较 #NOTE Q_svc_input 需要在外部输入
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
    @constraint(step_model, X_line .<= a) #NOTE a 需要在外部输入  这个仅在env.rest才需要

    #  4. 闭合的边数=总节点数-带电孤岛数-不带电孤立节点数
    @constraint(step_model, sum(b, dims=1) .== N_Bus .- sum(X_BS, dims=1) .- sum(ones(size(X_EN)) .- X_EN, dims=1))

    # % 线路操作约束
    @constraint(step_model, X_tieline .== X_tieline_input) # % 不需要下面的约束，直接从外部输入 #NOTE X_tieline_input 需要在外部输入

    # Obj
    @objective(step_model, Min, -sum(Pd_rec[:]) - 0.01*sum(X_line[:]) + 0.01*sum(e_Qsvc_up .+ e_Qsvc_down))

    return step_model
end

function make_opf(args_expert, args_step)
    global model = OPF_Core(args_expert, args_step)
    set_optimizer(model.expert_model, Gurobi.Optimizer) # 需要指定求解器
    set_optimizer(model.step_model, Gurobi.Optimizer)
    optimize!(model.expert_model)
    # res_v = value.(model.expert_model[:V])
    return objective_value(model.expert_model)
end

function opf()
    global model
    optimize!(model.expert_model)
    # res_v = value.(model.expert_model[:V])
    return objective_value(model.expert_model)
end

function mod_a(a_input)
    global model
    for i in 1:size(a_input,1)
        for j in 1:size(a_input,2)
            fix(model.expert_model[:a][i,j], a_input[i,j]) # 注意提取变量的方法需要[:变量名]
        end
    end
    optimize!(model.expert_model)
    return objective_value(model.expert_model)
end




