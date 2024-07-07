# 这个文件演示了如何在python中调用julia的函数
# 首先调用julia脚本建立基于JuMP的优化模型并求解
# 然后通过python给出新的约束条件，再次求解

from juliacall import Main as jl

jl.include("lp.jl") # 求解优化问题，并打印问题

b = -1 # python给出的新约束条件参数
c = 2
d = 0

jl.model_add(b,c,d) # 添加由b给出的新约束条件，求解，并打印问题