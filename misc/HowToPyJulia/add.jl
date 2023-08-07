function model_add(b,c,d)
    @constraint(model, b*x + c*y <= d)
    # 求解线性规划问题
    optimize!(model)
    
    # 打印结果
    println("Optimal solution:")
    println("x = ", value(x))
    println("y = ", value(y))
    println("Objective value = ", objective_value(model)) 
    println(model)   

end
