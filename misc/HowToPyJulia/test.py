from juliacall import Main as jl

jl.include("jtest.jl") # 求解优化问题，并打印问题

# 多态函数
x = jl.foo(-1)

a,b,c = x
print(a,b,c)