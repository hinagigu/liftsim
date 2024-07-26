import pandas as pd

# 读取Excel文件
file_path = 'gene.xlsx'  # 替换为你的Excel文件路径
df = pd.read_excel(file_path)

# 获取列名
columns = df.columns

# 初始化一个空的列表来存储结果
result_list = []

# 遍历每一行
for index, row in df.iterrows():
    gene_name = row[columns[0]]
    dcm_values = []
    con_values = []
    
    # 遍历每一列
    for col in columns[1:]:
        if row[col] != 0:
            if col.startswith('DCM'):
                dcm_values.append(str(row[col]))
            elif col.startswith('Con'):
                con_values.append(str(row[col]))
    
    # 添加DCM行
    if dcm_values:
        result_list.append({'GeneName': f'{gene_name}/DCM', 'Values': ' '.join(dcm_values)})
    
    # 添加CON行
    if con_values:
        result_list.append({'GeneName': f'{gene_name}/Con', 'Values': ' '.join(con_values)})

# 将结果列表转换为DataFrame
result_df = pd.DataFrame(result_list)

# 保存结果到新的Excel文件
result_file_path = 'newgene.xlsx'  # 替换为你想要保存的文件路径
result_df.to_excel(result_file_path, index=False)

print(f'Reformatted data has been saved to {result_file_path}')