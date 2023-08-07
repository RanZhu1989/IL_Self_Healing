import XLSX

# 从 Excel 文件读取数据
Bus_data = XLSX.readtable("Case_33BW_Data.xlsx", "Bus")
Bus_data = hcat(Bus_data.data ...)

DG_data = XLSX.readtable("Case_33BW_Data.xlsx", "DG")
DG_data = hcat(DG_data.data ...)

Branch_data = XLSX.readtable("Case_33BW_Data.xlsx", "Branch")
Branch_data = hcat(Branch_data.data ...)

file = open("33bw_data.jl", "w")

# 写入矩阵变量的名称和赋值符号
write(file, "DN_Bus_Data = [\n")

# 写入每一行的数据
for row in eachrow(Bus_data)
    write(file, "   ")
    write(file, repr(row))
    write(file, ",\n")
end

write(file, "]")  # 结束矩阵的定义


# 插入两行空行
write(file, "\n\n")

# 写入第二个矩阵的变量名称和赋值符号
write(file, "DN_DG_Data = [\n")

# 写入第二个矩阵的数据
for row in eachrow(DG_data)
    write(file, "   ")
    write(file, repr(row))
    write(file, ",\n")
end

write(file, "]")  # 结束第二个矩阵的定义


# 插入两行空行
write(file, "\n\n")

# 写入第二个矩阵的变量名称和赋值符号
write(file, "DN_Branch_Data = [\n")

# 写入第二个矩阵的数据
for row in eachrow(Branch_data)
    write(file, "   ")
    write(file, repr(row))
    write(file, ",\n")
end

write(file, "]")  # 结束第二个矩阵的定义

close(file)