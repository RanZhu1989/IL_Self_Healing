using JuMP
using Gurobi


a = 2.0  # 假设外部传入的 a 值为 2.0
# 创建一个模型对象
model = Model(Gurobi.Optimizer)

# 定义决策变量
@variable(model, x >= 0)
@variable(model, y >= 0)

# 定义约束条件
@constraint(model, a * x + 2y >= 10)
@constraint(model, 3x - y <= 15)

# 定义目标函数
@objective(model, Min, 4x + 5y)

# 求解线性规划问题
optimize!(model)

# 打印结果
println("Optimal solution:")
println("x = ", value(x))
println("y = ", value(y))
println("Objective value = ", objective_value(model))
println(model)   

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

