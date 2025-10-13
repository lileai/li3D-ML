import os
import ast
import glob
import pandas as pd
from tqdm import tqdm

def read_txt_dict(txt_data_path):
    # 获取所有txt文件
    file_names = glob.glob(f'{txt_data_path}/*.txt', recursive=True)
    for i, file_name in enumerate(file_names):
        print(f"[{i}]-{file_name}")
    file_idx = int(input("选择筛选的txt文件>>"))

    try:
        # 打开文件并逐行读取
        with open(file_names[file_idx], 'r', encoding='utf-8') as file:
            lines = file.readlines()  # 读取所有行

        # 用于存储解析后的字典
        txt_dicts = []

        # 遍历每一行，尝试解析为字典
        for line in lines:
            line = line.strip()  # 去掉首尾空格和换行符
            if not line:
                continue  # 跳过空行
            try:
                # 替换单引号为双引号（如果需要）
                line = line.replace("'", '"')
                # 使用 ast.literal_eval 安全地解析为字典
                txt_dict = ast.literal_eval(line)
                txt_dicts.append(txt_dict)  # 将解析后的字典添加到列表中
            except SyntaxError as e:
                print(f"解析错误：{e}，内容为：{line}")
            except Exception as e:
                print(f"其他错误：{e}，内容为：{line}")

    except FileNotFoundError:
        print(f"文件未找到，请检查路径：{file_names[file_idx]}")
    except Exception as e:
        print(f"发生错误：{e}")
    return txt_dicts

def read_xlsx_dict(xlsx_data_path):
    # 获取所有xlsx文件
    file_names = glob.glob(f'{xlsx_data_path}/*.xlsx', recursive=True)
    for i, file_name in enumerate(file_names):
        print(f"[{i}]-{file_name}")
    file_idx = int(input("选择筛选的xlsx文件>>"))

    df = pd.read_excel(file_names[file_idx])
    xlsx_dicts = df.to_dict(orient='records')
    xlsx_file_name = os.path.basename(file_names[file_idx])
    return xlsx_dicts, xlsx_file_name


def merge_data(xlsx_dicts, txt_dicts, txt_keys, xlsx_keys):
    """
    合并 xlsx 和 txt 数据。

    参数:
    - xlsx_dicts: Excel 数据的字典列表。
    - txt_dicts: TXT 数据的字典列表。
    - txt_keys: TXT 文件中用于匹配的键列表。
    - xlsx_keys: Excel 文件中用于匹配的键列表。

    返回:
    - merged_data: 合并后的数据列表。
    """
    # 创建一个空列表，用于存储合并后的数据
    merged_data = []

    # 遍历 xlsx_dicts 中的每一项
    for xlsx_item in tqdm(xlsx_dicts, desc="处理进度"):
        match_found = False  # 标记是否找到匹配项
        for xlsx_key in xlsx_keys:
            xlsx_value = xlsx_item.get(xlsx_key)
            if not xlsx_value:
                continue  # 如果没有对应的值，跳过当前项

            # 遍历 txt_dicts 中的每一项，查找匹配的键值
            for txt_item in txt_dicts:
                for txt_key in txt_keys:
                    txt_value = txt_item.get(txt_key)
                    if not txt_value:
                        continue  # 如果没有对应的值，跳过当前项

                    # 如果匹配成功
                    if xlsx_value in txt_value:
                        # 将匹配到的键值对添加到 xlsx_item 中
                        for key in txt_keys:
                            xlsx_item[f'txt_{key}'] = txt_item.get(key)
                        merged_data.append(xlsx_item)
                        match_found = True
                        break  # 匹配成功后跳出内层循环
                if match_found:
                    break  # 匹配成功后跳出外层循环

    return merged_data


if __name__ == '__main__':
    if not os.path.exists("../table"):
        os.makedirs("./table")
    # 定义文件路径
    output_excel_path = r'../table'  # Excel 文件路径
    txt_file_path = r'../table'  # 假设 txt 文件路径

    # 读取 xlsx 文件
    xlsx_dicts, xlsx_file_name = read_xlsx_dict(output_excel_path)

    # 读取 txt 文件
    txt_dicts = read_txt_dict(txt_file_path)

    # 定义匹配的键
    txt_keys = ['local_path', 'ship_det_uuid']  # TXT 文件中用于匹配的键
    xlsx_keys = ['Directory']  # Excel 文件中用于匹配的键

    # 合并数据
    merged_data = merge_data(xlsx_dicts, txt_dicts, txt_keys, xlsx_keys)

    # 将合并后的数据转换为 DataFrame
    merged_df = pd.DataFrame(merged_data)

    # 保存到新的 Excel 文件
    output_file_name = "../table/output.xlsx"
    output_file_path = os.path.join(output_excel_path, output_file_name)
    merged_df.to_excel(output_file_path, index=False)

    print(f"数据已成功合并到新的 Excel 文件中：{output_file_path}")

