using JuMP
using Gurobi

function test(args_expert)

    NT, N_Branch, N_TL, N_NL, N_Bus, pIn, N_DG, DG_Mask, R_Branch, X_Branch, Big_M_V, V0,
        V_min, V_max, Pd, Qd, S_Branch, P_DG_min, P_DG_max, Q_DG_min, Q_DG_max, BigM_SC, BSDG_Mask,
        Big_M_FF = args_expert
    
    test_model = Model(Gurobi.Optimizer)

    @variable(test_model, PF[1:N_Branch, 1:NT]) # active power flow at line ij
    @variable(test_model, QF[1:N_Branch, 1:NT]) # reactive power flow at line ij
    @variable(test_model, V[1:N_Bus, 1:NT]) # voltage at bus j

    @variable(test_model, P_dg[1:N_DG, 1:NT]) # 上游
    @variable(test_model, Q_dg[1:N_DG, 1:NT])

    @variable(test_model, b[1:N_Branch, 1:NT], Bin)

    # ------------------潮流--------------------
    # 1. Bus PQ Blance: S_jk - S_ij = S_inj
    @constraint(test_model, pIn * PF .== DG_Mask * P_dg .- Pd)  # 添加约束
    @constraint(test_model, pIn * QF .== DG_Mask * Q_dg .- Qd)

    # % 4. 线路
    @constraint(test_model, PF .>= -S_Branch .* b)
    @constraint(test_model, PF .<= S_Branch .* b)
    @constraint(test_model, QF .>= -S_Branch .* b)
    @constraint(test_model, QF .<= S_Branch .* b)

    # ------------DG ----------------
    @constraint(test_model, P_dg .>= P_DG_min) 
    @constraint(test_model, P_dg .<= P_DG_max)    
    @constraint(test_model, Q_dg .>= Q_DG_min)
    @constraint(test_model, Q_dg .<= Q_DG_max)
    @constraint(test_model, Q_dg[1,:] .== 0.019)
    @constraint(test_model, Q_dg[2,:] .== 0.002)
    @constraint(test_model, Q_dg[3,:] .== -0.002)
    @constraint(test_model, Q_dg[4,:] .== -0.002)
    @constraint(test_model, Q_dg[5:end,:] .== 0.002)


    @constraint(test_model, b[1:N_NL,:] .== 1)
    @constraint(test_model, b[N_NL+1:N_Branch,:] .== 0)

    @objective(test_model,Min,1)

    optimize!(test_model)
end
