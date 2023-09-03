# Core OPF models for SelfHealingEnv
using JuMP

struct OPF_Core
    """
    The core models are stored in the struct which contains three JuMP models:
        expert_model: the model used to generate expert plolicy for TESTING and imitation learning
        step_model: the model used to step the environment
        reset_model: the model used to reset the environment

    Work flow of the core:
    (sys. data input)            Init. Models                    set_dmg()                      Reset Env.
        Args_*  ---------->  | make_reset_model()  |  ------[Fix Var.==Para.]--------->| set_ & solve_ResetModel()  |
                             | make_expert_model() |                            |      |       paras.|              |
                             | make_step_model()   |                            |      |             V              |
                                                                                |      | set_ & solve_ExpertModel() |
                                                                                |       -----------------------------
                                                                                |                   |
                                                                                |            paras. |
                                                                                |                   V
                                                                                |               Step Env.
                                                                                |----->| set_ & solve_StepModel()   |

    Developing Notes: 
        To change parameters of the models, JuMP provides three ways including:
            1. Remove the old constraint and add a new one: JuMP.delete(model, con);
            2. Change the right hand side of the constraint: JuMP.set_normalized_rhs(con, new_rhs);
            3. Fix the variable: JuMP.fix(var, value).
        The first way can not work because it do not support creat constraints using in-built variables from external functions.
        The second way requires the normalized form, which is lack of universality.
        Although the third way could introduce intractable parts such as bilinear terms, there is no parameters multiplied by the variables.
        Therefore, it is a convenient way to change the parameters of the models. 
    """
    expert_model::JuMP.Model
    step_model::JuMP.Model
    reset_model::JuMP.Model
    
    function OPF_Core(
        args_expert::Tuple{Int64, Int64, Int64, Int64, Int64, Matrix{Float64}, Int64,
                            Matrix{Float64},Matrix{Float64}, Matrix{Float64}, Int64, 
                            Float64, Float64, Float64, Matrix{Float64}, Matrix{Float64}, 
                            Matrix{Float64}, Matrix{Float64}, Matrix{Float64}, Matrix{Float64}, 
                            Matrix{Float64}, Int64, Matrix{Float64}, Int64},

        args_step::Tuple{Int64, Int64, Int64, Int64, Int64, Matrix{Float64}, Int64, 
                            Matrix{Float64},Vector{Float64}, Vector{Float64}, Int64, Float64, 
                            Float64, Float64, Vector{Float64},Vector{Float64}, Vector{Float64}, 
                            Vector{Float64}, Vector{Float64}, Vector{Float64},Vector{Float64}, Matrix{Float64}, Int64}
    )::Nothing
        expert_model = make_expert_model(args_expert)
        step_model = make_step_model(args_step)
        reset_model = make_reset_model(args_step)
        new(expert_model, step_model, reset_model)
    end

end

"""------------------------------ Model Definitions ----------------------------------"""

function make_expert_model(args_expert)::JuMP.Model

    NT, N_Branch, N_TL, N_NL, N_Bus, pIn, N_DG, DG_Mask, R_Branch, X_Branch, Big_M_V, V0,
    V_min, V_max, Pd, Qd, S_Branch, P_DG_min, P_DG_max, Q_DG_min, Q_DG_max, BigM_SC, BSDG_Mask,
    Big_M_FF = args_expert
    
    expert_model = Model()

    #--Variables--
    # Fake Variables waiting to be fixed
    @variable(expert_model, a[1:N_NL, 1:NT]) # Damaged lines <- set_dmg()
    @variable(expert_model, X_tieline0[1:N_TL]) # TieLine line state at S0 <- set_ResetModel()
    @variable(expert_model, X_rec0[1:N_Bus]) # Load pick up at S0 <- set_ResetModel() <- solve_ResetModel
    @variable(expert_model, X_line0[1:N_NL]) # Non-tieline line state at S0 <- set_ResetModel() <- solve_ResetModel

    # p.u. variables
    @variable(expert_model, PF[1:N_Branch, 1:NT]) # Active power flow at line ij
    @variable(expert_model, QF[1:N_Branch, 1:NT]) # Reactive power flow at line ij
    @variable(expert_model, V[1:N_Bus, 1:NT]) # Voltage at bus j
    @variable(expert_model, P_dg[1:N_DG, 1:NT]) # Power output of DG
    @variable(expert_model, Q_dg[1:N_DG, 1:NT])
    @variable(expert_model, delta_Qdg[1:N_DG-1, 1:NT-1]) # Reactive power change value
    @variable(expert_model, Pd_rec[1:N_Bus, 1:NT]) # Recovered load
    @variable(expert_model, Qd_rec[1:N_Bus, 1:NT])
    @variable(expert_model, FF[1:N_Branch, 1:NT]) # commodity flow at line ij

    # binary variables
    @variable(expert_model, X_rec[1:N_Bus, 1:NT], Bin) # Load pick up
    @variable(expert_model, X_EN[1:N_Bus, 1:NT], Bin) # Node energized
    @variable(expert_model, X_tieline[1:N_TL, 1:NT], Bin) # Tieline status
    @variable(expert_model, X_line[1:N_NL, 1:NT], Bin) # Non-tieline status
    @variable(expert_model, z_bs[1:N_Bus, 1:NT], Bin) # For bilinear term
    @variable(expert_model, b[1:N_Branch, 1:NT], Bin) # Line energized status
    @variable(expert_model, X_BS[1:N_Bus, 1:NT], Bin) # Node with black start capability
    @variable(expert_model, z_bs1[1:N_Bus, 1:NT], Bin) # For logic constraint
    @variable(expert_model, z_bs2[1:N_Bus, 1:NT], Bin)
    @variable(expert_model, z_dg[1:N_DG-1, 1:NT-1], Bin) # For regulazation of DG output

    # --Constraints--
    # Power flow bus PQ blance
    @constraint(expert_model, pIn * PF .== DG_Mask * P_dg .- Pd_rec)
    @constraint(expert_model, pIn * QF .== DG_Mask * Q_dg .- Qd_rec)

    # Power flow voltage
    @constraint(expert_model, V0 .* (pIn' * V) .- R_Branch .* PF .- X_Branch .* QF .<= Big_M_V .* (1 .- b))
    @constraint(expert_model, V0 .* (pIn' * V) .- R_Branch .* PF .- X_Branch .* QF .>= -Big_M_V .* (1 .- b))
    @constraint(expert_model, V0 .* X_BS .+ X_EN .* V_min .- z_bs .* V_min .<= V)
    @constraint(expert_model, V .<= X_BS .* V0 .+ X_EN .* V_max .- z_bs .* V_max)
    @constraint(expert_model, z_bs .<= X_BS)
    @constraint(expert_model, z_bs .<= X_EN)
    @constraint(expert_model, z_bs .>= X_BS .+ X_EN .- 1)

    # Power flow load curtailments
    @constraint(expert_model, X_rec .<= X_EN)
    @constraint(expert_model, X_rec[1,:] .== 0)
    @constraint(expert_model, Pd_rec .== X_rec .* Pd)
    @constraint(expert_model, Qd_rec .== X_rec .* Qd)
    @constraint(expert_model, X_rec[:,2:NT] .>= X_rec[:,1:NT-1])
    @constraint(expert_model, X_rec[:,1] .>= X_rec0) #NOTE: Requiring fixing X_rec0

    # Power flow thermal limits
    @constraint(expert_model, PF .>= -S_Branch .* b)
    @constraint(expert_model, PF .<= S_Branch .* b)
    @constraint(expert_model, QF .>= -S_Branch .* b)
    @constraint(expert_model, QF .<= S_Branch .* b)

    # DG
    @constraint(expert_model, P_dg .>= (DG_Mask'*X_EN) .* P_DG_min) 
    @constraint(expert_model, P_dg .<= (DG_Mask'*X_EN) .* P_DG_max)
    @constraint(expert_model, Q_dg .>= (DG_Mask'*X_EN) .* Q_DG_min)
    @constraint(expert_model, Q_dg .<= (DG_Mask'*X_EN) .* Q_DG_max)
    @constraint(expert_model, BigM_SC .* (1 .- z_dg) .<= Q_dg[2:N_DG,2:NT] .- Q_dg[2:N_DG,1:NT-1])
    @constraint(expert_model, -BigM_SC .* (1 .- z_dg) .<= delta_Qdg .- (Q_dg[2:N_DG,2:NT] .- Q_dg[2:N_DG,1:NT-1]))
    @constraint(expert_model, BigM_SC .* (1 .- z_dg) .>= delta_Qdg .- (Q_dg[2:N_DG,2:NT] .- Q_dg[2:N_DG,1:NT-1]))
    @constraint(expert_model, -BigM_SC .* z_dg .<= delta_Qdg .+ (Q_dg[2:N_DG,2:NT] .- Q_dg[2:N_DG,1:NT-1]))
    @constraint(expert_model, BigM_SC .* z_dg .>= delta_Qdg .+ (Q_dg[2:N_DG,2:NT] .- Q_dg[2:N_DG,1:NT-1]))

    # Commodity flow based topology logic
    @constraint(expert_model, X_BS .== repeat(sum(BSDG_Mask,dims=2),1,NT))
    @constraint(expert_model, pIn * FF .+ X_EN .<= Big_M_FF .* (1 .- z_bs1))
    @constraint(expert_model, pIn * FF .+ X_EN .>= -Big_M_FF .* (1 .- z_bs1))
    @constraint(expert_model, z_bs1 .- 1 .<= X_BS)
    @constraint(expert_model, X_BS .<= 1 .- z_bs1)
    @constraint(expert_model, pIn * FF .>= -Big_M_FF .* (1 .- z_bs2))
    @constraint(expert_model, z_bs2 .- 1 .<= X_BS .- 1)
    @constraint(expert_model, X_BS .- 1 .<= 1 .- z_bs2)
    @constraint(expert_model, X_EN .- X_BS .>= -Big_M_FF .* (1 .- z_bs2))
    @constraint(expert_model, X_EN .- X_BS .<= Big_M_FF .* (1 .- z_bs2))
    @constraint(expert_model, z_bs1 .+ z_bs2 .== 1)
    @constraint(expert_model, -Big_M_FF .* b .<= FF)
    @constraint(expert_model, FF .<= Big_M_FF .* b)
    @constraint(expert_model, b .== [X_line; X_tieline])
    @constraint(expert_model, X_line .<= a ) #NOTE: Requiring fixing a
    @constraint(expert_model, sum(b, dims=1) .== N_Bus .- sum(X_BS, dims=1) .- sum(1 .- X_EN, dims=1))

    # TieLine action logic
    @constraint(expert_model, X_tieline[:, 2:NT] .>= X_tieline[:, 1:NT-1])
    @constraint(expert_model, X_tieline[:, 1] .>= X_tieline0) #NOTE: Requiring fixing X_tieline0
    @constraint(expert_model, sum(X_tieline[:, 2:NT] .- X_tieline[:, 1:NT-1], dims=1) .<= 1)
    @constraint(expert_model, sum(X_tieline[:, 1] .- X_tieline0, dims=1) .<= 1) #NOTE: Requiring fixing X_tieline0
    # The following two constraints are added to avoid infeasible solutions since non-tielines can not be operated
    @constraint(expert_model, X_line[:,2:NT] .>= X_line[:,1:NT-1])
    @constraint(expert_model, X_line[:,1] .>= X_line0) #NOTE: Requiring fixing X_line0

    # Obj
    @objective(expert_model, Min, -sum(Pd_rec[:]) - 0.01*sum(X_line[:]) + sum(delta_Qdg[:]))

    return expert_model  
end

function make_step_model(args_step)::JuMP.Model

    NT, N_Branch, N_TL, N_NL, N_Bus, pIn, N_DG, DG_Mask, R_Branch, X_Branch, Big_M_V, V0,
    V_min, V_max, Pd, Qd, S_Branch, P_DG_min, P_DG_max, Q_DG_min,
    Q_DG_max, BSDG_Mask, Big_M_FF = args_step

    step_model = Model()

    # --Variables--
    # Fake variables waiting to be fixed
    @variable(step_model, a[1:N_NL]) # Damaged lines <- set_dmg()
    @variable(step_model, X_rec0[1:N_Bus]) # Load pick up at last step <- set_StepModel() <- solve_ResetModel\StepModel()
    @variable(step_model, Q_svc[1:N_DG-1]) # Reactive power command <- set_StepModel()
    @variable(step_model, X_tieline[1:N_TL, 1:NT], Bin) # Tieline action <- set_StepModel()

    # p.u. variables
    @variable(step_model, PF[1:N_Branch, 1:NT])
    @variable(step_model, QF[1:N_Branch, 1:NT])
    @variable(step_model, V[1:N_Bus, 1:NT])
    @variable(step_model, P_dg[1:N_DG, 1:NT])
    @variable(step_model, Q_dg[1:N_DG, 1:NT])
    @variable(step_model, e_Qsvc[1:N_DG-1]) # Tracking error of Q_svc command
    @variable(step_model, e_Qsvc_up[1:N_DG-1]) # For ABS():= e_Qsvc_up + e_Qsvc_down
    @variable(step_model, e_Qsvc_down[1:N_DG-1]) 
    @variable(step_model, Pd_rec[1:N_Bus, 1:NT])
    @variable(step_model, Qd_rec[1:N_Bus, 1:NT])
    @variable(step_model, FF[1:N_Branch, 1:NT])

    # binary variables
    @variable(step_model, X_rec[1:N_Bus, 1:NT], Bin)
    @variable(step_model, X_EN[1:N_Bus, 1:NT], Bin)
    @variable(step_model, X_line[1:N_NL, 1:NT], Bin)
    @variable(step_model, z_bs[1:N_Bus, 1:NT], Bin)
    @variable(step_model, b[1:N_Branch, 1:NT], Bin)
    @variable(step_model, X_BS[1:N_Bus, 1:NT], Bin)
    @variable(step_model, z_bs1[1:N_Bus, 1:NT], Bin)
    @variable(step_model, z_bs2[1:N_Bus, 1:NT], Bin)

    # --Constraints--
    # Power flow bus PQ blance
    @constraint(step_model, pIn * PF .== DG_Mask * P_dg .- Pd_rec)
    @constraint(step_model, pIn * QF .== DG_Mask * Q_dg .- Qd_rec)

    # Power flow voltage
    @constraint(step_model, V0 .* (pIn' * V) .- R_Branch .* PF .- X_Branch .* QF .<= Big_M_V .* (1 .- b))
    @constraint(step_model, V0 .* (pIn' * V) .- R_Branch .* PF .- X_Branch .* QF .>= -Big_M_V .* (1 .- b))
    @constraint(step_model, V0 .* X_BS .+ X_EN .* V_min .- z_bs .* V_min .<= V)
    @constraint(step_model, V .<= X_BS .* V0 .+ X_EN .* V_max .- z_bs .* V_max)
    @constraint(step_model, z_bs .<= X_BS)
    @constraint(step_model, z_bs .<= X_EN)
    @constraint(step_model, z_bs .>= X_BS .+ X_EN .- 1)

    # Power flow load curtailments
    @constraint(step_model, X_rec .<= X_EN)
    @constraint(step_model, X_rec[1,:] .== 0)
    @constraint(step_model, Pd_rec .== X_rec .* Pd)
    @constraint(step_model, Qd_rec .== X_rec .* Qd)
    @constraint(step_model, X_rec[:,1] .>= X_rec0) #NOTE: Requiring fixing X_rec0

    # Power flow thermal limits
    @constraint(step_model, PF .>= -S_Branch .* b)
    @constraint(step_model, PF .<= S_Branch .* b)
    @constraint(step_model, QF .>= -S_Branch .* b)
    @constraint(step_model, QF .<= S_Branch .* b)

    # DG
    @constraint(step_model, P_dg .>= (DG_Mask'*X_EN) .* P_DG_min)
    @constraint(step_model, P_dg .<= (DG_Mask'*X_EN) .* P_DG_max)
    @constraint(step_model, Q_dg .>= (DG_Mask'*X_EN) .* Q_DG_min)
    @constraint(step_model, Q_dg .<= (DG_Mask'*X_EN) .* Q_DG_max)
    @constraint(step_model, e_Qsvc .== Q_svc .- Q_dg[2:N_DG,1] ) #NOTE: Requiring fixing Q_svc
    @constraint(step_model, e_Qsvc .== e_Qsvc_up .- e_Qsvc_down)
    @constraint(step_model, e_Qsvc_up .>= 0)
    @constraint(step_model, e_Qsvc_down .>= 0)

    # Commodity flow based topology logic
    @constraint(step_model, X_BS .== repeat(sum(BSDG_Mask,dims=2),1,NT))
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
    @constraint(step_model, -Big_M_FF .* b .<= FF)
    @constraint(step_model, FF .<= Big_M_FF .* b)
    @constraint(step_model, b .== [X_line; X_tieline])
    @constraint(step_model, X_line .<= a) #NOTE: Requiring fixing a
    @constraint(step_model, sum(b, dims=1) .== N_Bus .- sum(X_BS, dims=1) .- sum(1 .- X_EN, dims=1))

    # Obj
    @objective(step_model, Min, -sum(Pd_rec[:]) - 0.01*sum(X_line[:]) + 1000*sum(e_Qsvc_up .+ e_Qsvc_down))

    return step_model
end

function make_reset_model(args_step)::JuMP.Model

    NT, N_Branch, N_TL, N_NL, N_Bus, pIn, N_DG, DG_Mask, R_Branch, X_Branch, Big_M_V, V0,
    V_min, V_max, Pd, Qd, S_Branch, P_DG_min, P_DG_max, Q_DG_min,
    Q_DG_max, BSDG_Mask, Big_M_FF = args_step

    reset_model = Model()

    #--Variables--
    # Fake Variables waiting to be fixed
    @variable(reset_model, a[1:N_NL]) # Damaged lines <- set_dmg()
    @variable(reset_model, Q_svc[1:N_DG-1]) # Reactive power command at S0 <- set_ResetModel()
    @variable(reset_model, X_tieline[1:N_TL, 1:NT], Bin) # Tieline action at S0 <- set_ResetModel()

    # p.u. variables
    @variable(reset_model, PF[1:N_Branch, 1:NT])
    @variable(reset_model, QF[1:N_Branch, 1:NT])
    @variable(reset_model, V[1:N_Bus, 1:NT])
    @variable(reset_model, P_dg[1:N_DG, 1:NT])
    @variable(reset_model, Q_dg[1:N_DG, 1:NT])
    @variable(reset_model, Pd_rec[1:N_Bus, 1:NT])
    @variable(reset_model, Qd_rec[1:N_Bus, 1:NT])
    @variable(reset_model, FF[1:N_Branch, 1:NT])

    # binary variables
    @variable(reset_model, X_rec[1:N_Bus, 1:NT], Bin)
    @variable(reset_model, X_EN[1:N_Bus, 1:NT], Bin)
    @variable(reset_model, X_line[1:N_NL, 1:NT], Bin)
    @variable(reset_model, z_bs[1:N_Bus, 1:NT], Bin)
    @variable(reset_model, b[1:N_Branch, 1:NT], Bin)
    @variable(reset_model, X_BS[1:N_Bus, 1:NT], Bin)
    @variable(reset_model, z_bs1[1:N_Bus, 1:NT], Bin)
    @variable(reset_model, z_bs2[1:N_Bus, 1:NT], Bin)

    # --Constraints--
    # Power flow bus PQ blance
    @constraint(reset_model, pIn * PF .== DG_Mask * P_dg .- Pd_rec)
    @constraint(reset_model, pIn * QF .== DG_Mask * Q_dg .- Qd_rec)

    # Power flow voltage
    @constraint(reset_model, V0 .* (pIn' * V) .- R_Branch .* PF .- X_Branch .* QF .<= Big_M_V .* (1 .- b))
    @constraint(reset_model, V0 .* (pIn' * V) .- R_Branch .* PF .- X_Branch .* QF .>= -Big_M_V .* (1 .- b))
    @constraint(reset_model, V0 .* X_BS .+ X_EN .* V_min .- z_bs .* V_min .<= V)
    @constraint(reset_model, V .<= X_BS .* V0 .+ X_EN .* V_max .- z_bs .* V_max)
    @constraint(reset_model, z_bs .<= X_BS)
    @constraint(reset_model, z_bs .<= X_EN)
    @constraint(reset_model, z_bs .>= X_BS .+ X_EN .- 1)

    # Power flow load curtailments
    @constraint(reset_model, X_rec .<= X_EN)
    @constraint(reset_model, X_rec[1,:] .== 0)
    @constraint(reset_model, Pd_rec .== X_rec .* Pd)
    @constraint(reset_model, Qd_rec .== X_rec .* Qd)

    # Power flow thermal limits
    @constraint(reset_model, PF .>= -S_Branch .* b)
    @constraint(reset_model, PF .<= S_Branch .* b)
    @constraint(reset_model, QF .>= -S_Branch .* b)
    @constraint(reset_model, QF .<= S_Branch .* b)

    # DG
    @constraint(reset_model, P_dg .>= (DG_Mask'*X_EN) .* P_DG_min) # 这里要再调度一下
    @constraint(reset_model, P_dg .<= (DG_Mask'*X_EN) .* P_DG_max)
    @constraint(reset_model, Q_dg .>= (DG_Mask'*X_EN) .* Q_DG_min)
    @constraint(reset_model, Q_dg .<= (DG_Mask'*X_EN) .* Q_DG_max)
    
    @constraint(reset_model, Q_dg[2:N_DG,1] .== Q_svc) #  #NOTE: Requiring fixing Q_svc

    # Commodity flow based topology logic
    @constraint(reset_model, X_BS .== repeat(sum(BSDG_Mask,dims=2),1,NT))
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
    @constraint(reset_model, -Big_M_FF .* b .<= FF)
    @constraint(reset_model, FF .<= Big_M_FF .* b)
    @constraint(reset_model, b .== [X_line; X_tieline]) #NOTE: Requiring fixing X_tieline
    @constraint(reset_model, X_line .<= a) #NOTE: Requiring fixing a
    @constraint(reset_model, sum(b, dims=1) .== N_Bus .- sum(X_BS, dims=1) .- sum(1 .- X_EN, dims=1))

    # Obj
    @objective(reset_model, Min, -sum(Pd_rec[:]) - 0.01*sum(X_line[:]))

    return reset_model
end

"""------------------------------ End of Definitions ----------------------------------"""



"""------------------------------ Utility Functions ----------------------------------
Servel utility functions are defined for managing the core in an external way:

    init_opf_core: Function as __init__() for initializing the struct
    set_dmg: Set the damaged lines for ALL three models
    set_*Model: Transfer parameters to the corresponding Model
    solve_*Model: Solve the corresponding Model
"""

function init_opf_core(; 
    args_expert::Tuple{Int64, Int64, Int64, Int64, Int64, Matrix{Float64}, Int64,
                        Matrix{Float64},Matrix{Float64}, Matrix{Float64}, Int64, 
                        Float64, Float64, Float64, Matrix{Float64}, Matrix{Float64}, 
                        Matrix{Float64}, Matrix{Float64}, Matrix{Float64}, Matrix{Float64}, 
                        Matrix{Float64}, Int64, Matrix{Float64}, Int64},
    args_step::Tuple{Int64, Int64, Int64, Int64, Int64, Matrix{Float64}, Int64, 
                        Matrix{Float64},Vector{Float64}, Vector{Float64}, Int64, Float64, 
                        Float64, Float64, Vector{Float64},Vector{Float64}, Vector{Float64}, 
                        Vector{Float64}, Vector{Float64}, Vector{Float64},Vector{Float64}, Matrix{Float64}, Int64},
    solver::String="CPLEX",
    MIP_gap_expert_model::Float64=1e-4,
    MIP_gap_step_model::Float64=1e-4,
    MIP_gap_reset_model::Float64=1e-4,
    display::Bool=true
)::Nothing
    """
    We can not use jl.OPF_Core() thourgh JuliaPy interface. We use this function to initialize the core.
    
    Args:

        args_expert: A tuple of parameters for expert model
        args_step: A tuple of parameters for step model
        solver: Solver used for solving the models
        MIP_gap_expert_model: The MIP gap for expert model
        MIP_gap_step_model: The MIP gap for step model
        MIP_gap_reset_model: The MIP gap for reset model
        display: Whether to display the solving process
    """

    # Use global variable to avoid using eval() in JuliaPy 
    global core = OPF_Core(args_expert, args_step)

    if solver == "CPLEX"
        #NOTE: You can add more attributes
        expert_model_optimizer = optimizer_with_attributes(
            CPLEX.Optimizer, "CPXPARAM_ScreenOutput" => display, "CPX_PARAM_EPGAP" => MIP_gap_expert_model
        )
        step_model_optimizer = optimizer_with_attributes(
            CPLEX.Optimizer, "CPXPARAM_ScreenOutput" => display, "CPX_PARAM_EPGAP" => MIP_gap_step_model
        )
        reset_model_optimizer = optimizer_with_attributes(
            CPLEX.Optimizer, "CPXPARAM_ScreenOutput" => display, "CPX_PARAM_EPGAP" => MIP_gap_reset_model
        )
    elseif solver == "Gurobi"
        expert_model_optimizer = optimizer_with_attributes(
            Gurobi.Optimizer, "output_flag" => display, "MIPGap" => MIP_gap_expert_model
        )
        step_model_optimizer = optimizer_with_attributes(
            Gurobi.Optimizer, "output_flag" => display, "MIPGap" => MIP_gap_step_model
        )
        reset_model_optimizer = optimizer_with_attributes(
            Gurobi.Optimizer, "output_flag" => display, "MIPGap" => MIP_gap_reset_model
        )
    else
        #NOTE: Add more solvers if needed
        error("solver not supported")
    end

    set_optimizer(core.expert_model, expert_model_optimizer)
    set_optimizer(core.step_model, step_model_optimizer)
    set_optimizer(core.reset_model, reset_model_optimizer)

end

function set_dmg(a_input::Matrix{Float64})::Nothing
    """
    Set the damaged lines for ALL three models. 
    In each episode, this function should be called ONLY ONE time when the disturbances are determined.
    """
    global core

    for idx in eachindex(a_input)
        fix(core.expert_model[:a][idx], a_input[idx])
    end

    for idx in eachindex(a_input[:,1])
        fix(core.step_model[:a][idx], a_input[idx,1])
        fix(core.reset_model[:a][idx], a_input[idx,1])
    end

end

function set_ExpertModel(;
    X_tieline0_input::Vector{Float64},
    X_rec0_input::Vector{Int8},
    X_line0_input::Vector{Int8},
    vvo::Bool=true
)::Nothing
    """
    Set the initial state for expert model. The inputs are partly from the results of rest model.
    """
    global core

    for idx in eachindex(X_tieline0_input)
        fix(core.expert_model[:X_tieline0][idx],X_tieline0_input[idx])
    end

    for idx in eachindex(X_rec0_input)
        fix(core.expert_model[:X_rec0][idx],X_rec0_input[idx])
    end

    for idx in eachindex(X_line0_input)
        fix(core.expert_model[:X_line0][idx], X_line0_input[idx])
    end

    # Disable SVC in non-vvo model 
    if !vvo
        for idx in eachindex(core.expert_model[:Q_dg][2:end,:])
            fix(core.expert_model[:Q_dg][2:end,:][idx],0)
        end
    end
end

function set_StepModel(; 
    X_rec0_input::Vector{Int8},
    X_tieline_input::Vector{Int8},
    Q_svc_input::Union{Vector{Float64},Nothing} =nothing,
    vvo::Bool=true
)::Nothing
    """
    Set the initial state for step model. The inputs are partly from the results of rest model or the pervious step model.
    """

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
        if vvo
            fix(core.step_model[:Q_svc][idx],Q_svc_input[idx])
        else
            fix(core.step_model[:Q_svc][idx],0)
        end
    end
    
end

function set_ResetModel(; 
    X_tieline_input::Vector{Float64}, 
    Q_svc_input::Vector{Float64}
)::Nothing
    """
    Set the parameters for reset model.
    """

    global core

    for idx in eachindex(X_tieline_input)
        fix(core.reset_model[:X_tieline][idx],X_tieline_input[idx])
    end

    for idx in eachindex(Q_svc_input)
        fix(core.reset_model[:Q_svc][idx],Q_svc_input[idx])
    end
    
end

function solve_ExpertModel()::Tuple{Bool, Union{Nothing, Matrix{Float64}}, Union{Nothing, Matrix{Float64}}, Union{Nothing, Matrix{Float64}}, 
                                    Union{Nothing, Matrix{Float64}}, Union{Nothing, Matrix{Float64}}, Union{Nothing, Matrix{Float64}}}
    """
    Solve the expert model and return the results. The model can be infeasible.
    """
    global core
    optimize!(core.expert_model)
    solved_flag = false

    if termination_status(core.expert_model) == MOI.OPTIMAL
        solved_flag = true
        b = value.(core.expert_model[:b])
        x_tieline = value.(core.expert_model[:X_tieline])
        x_load = value.(core.expert_model[:X_rec])
        Pg = value.(core.expert_model[:P_dg])
        Qg = value.(core.expert_model[:Q_dg])
        Prec = sum(value.(core.expert_model[:Pd_rec]),dims=1)

        return solved_flag, b, x_tieline, x_load, Pg, Qg, Prec
    else 
        return solved_flag, nothing, nothing, nothing, nothing, nothing, nothing
    end

end

function solve_StepModel()::Tuple{Bool, Union{Nothing, Vector{Float64}}, Union{Nothing, Vector{Float64}}, 
                                    Union{Nothing, Vector{Float64}}, Union{Nothing, Vector{Float64}}, 
                                    Union{Nothing, Vector{Float64}}, Union{Nothing, Float64}, Union{Nothing, Float64}}
    """
    Solve the step model and return the results. The model can be infeasible.
    """
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
        e_Qsvc = value.(sum(core.step_model[:e_Qsvc_up] .+ core.step_model[:e_Qsvc_down])) # Total tracking error
        
        return solved_flag, b, x_tieline, x_load, PF, QF, Prec, e_Qsvc
        
    else
        return solved_flag, nothing, nothing, nothing, nothing, nothing, nothing, nothing 
    end
end

function solve_ResetModel()::Tuple{Vector{Float64}, Vector{Float64}, Vector{Float64}, 
                                    Vector{Float64}, Vector{Float64}, Float64}
    """
    Solve the reset model and return the results. The model is always feasible.
    """
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