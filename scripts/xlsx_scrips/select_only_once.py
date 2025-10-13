import pandas as pd
from datetime import datetime
# 读取 Excel 文件
directory_path = '../table'
file_path = f'{directory_path}/output.xlsx'  # 替换为你的文件路径
sheet_name = 'Sheet1'         # 替换为你的工作表名称
df = pd.read_excel(file_path, sheet_name=sheet_name)

# 假设我们要筛选的列名为 'Column_Name'
column_name = 'txt_ship_det_uuid'  # 替换为你的列名

# 统计每个条目的出现次数
value_counts = df[column_name].value_counts()

# 筛选出只出现一次的条目
unique_entries = value_counts[value_counts == 1].index.tolist()

# 打印结果
print("只出现一次的条目：")
print(unique_entries)

# 获取当前时间戳
current_time = datetime.now().strftime("%Y%m%d")  # 格式化为年月日_时分秒
# 如果需要，可以将这些条目保存到新的 Excel 文件中
output_file = f'{directory_path}/{current_time}_unique_entries.xlsx'  # 输出文件路径
unique_df = df[df[column_name].isin(unique_entries)]
unique_df.to_excel(output_file, index=False)

print(f"筛选结果已保存到 {output_file}")