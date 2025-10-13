import pandas as pd
from datetime import datetime
# 读取两张Excel表
df1 = pd.read_excel('../table/output.xlsx')  # 包含id的表
df2 = pd.read_excel('../table/真实值.xlsx')  # 包含id信息和其他的表

# 对表2进行去重操作，保留第一个出现的行
# 假设表2中重复的搜索词对应的值是一样的，去重不会丢失信息
df2_unique = df2.drop_duplicates(subset='搜索词', keep='first')

# 查找匹配的Directory（表1中的id）和搜索词（表2中的id），
# 并提取“真实干舷”、“水位计”和“水位计补偿”列
# 使用merge函数，基于Directory和搜索词进行左连接
merged_df = pd.merge(df1, df2_unique[['搜索词', '真实干舷', '水位计', '水位计补偿']],
                     left_on='txt_ship_det_uuid', right_on='搜索词', how='left')

# 如果需要，可以删除多余的“搜索词”列
merged_df.drop(columns=['搜索词'], inplace=True)
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")  # 格式化为年月日_时分秒
# 保存结果到新的Excel文件
merged_df.to_excel(f'../table/{current_time}_结果.xlsx', index=False)

print(f"处理完成，结果已保存到'{current_time}_结果.xlsx'")