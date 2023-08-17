# 保存矩阵到文件

# 内存中的矩阵
A = [1 2 3; 4 5 6; 7 8 9]

B = [1 2 13; 4 5 16; 722 8 9]

# 每行元素的空格数和行与行之间的空格数
elementSpaces = 4  # 每个元素之间的空格数量
elementSpaces = 5    # 每行之间的空格数量
emptyLines = 1  # 每行之间的空行数量
# 将矩阵保存到文件




function write_data(input_matrix, matrix_name, file, elementSpaces=4, lineSpaces=4, emptyLines=1)
    """
    input_matrix: 输入的矩阵
    matrix_name: 文件中保存矩阵的名称
    file: 文件流
    elementSpaces: 每个元素之间的空格数量
    lineSpaces: 每行到开头的空格数量
    emptyLines: 每行之间的空行数量
    """
    write(file, "$matrix_name = [\n")
    for i in 1:size(input_matrix, 1)
        write(file, " " ^ lineSpaces)  # 行与行之间的空格
        for j in 1:size(input_matrix, 2)
            write(file, string(input_matrix[i, j]))
            if j < size(input_matrix, 2)
                write(file, " " ^ elementSpaces)  # 元素之间的空格
            end
        end
        if i < size(input_matrix, 1)
            write(file, "\n" ^ emptyLines)  # 空行
        end
    end
    write(file, "]")
    write(file,"\n\n")
end




file = open("33bw_data.jl", "w")

write_data(A,"aaa",file)
write_data(B,"bbb",file)

close(file)