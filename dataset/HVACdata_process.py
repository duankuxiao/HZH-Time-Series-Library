import os
import pandas as pd

def merge_data():
    # 设置你的Excel文件所在路径
    folder_path = r'D:\Time-LLM-main\dataset\HVAC\summary'  # 替换为你的文件夹路径
    output_path = os.path.join(r'D:\Time-LLM-main\dataset\HVAC','summary.xlsx')

    # 用于存储所有 sheet 的数据
    sheet_data = {}

    # 遍历文件夹中的所有 .xlsx 文件
    for filename in os.listdir(folder_path):
        if filename.endswith('.xlsx'):
            file_path = os.path.join(folder_path, filename)
            try:
                xls = pd.ExcelFile(file_path, engine='openpyxl')
            except Exception as e:
                print(f"❌ 无法读取文件 {filename}，跳过。原因：{e}")
                continue

            for sheet_name in xls.sheet_names:
                try:
                    df = xls.parse(sheet_name, header=1)  # 第2行作为列名
                    df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0], errors='coerce')  # 第一列转为时间
                    df = df.dropna(subset=[df.columns[0]])  # 删除无法解析为时间的行
                    df = df.set_index(df.columns[0])  # 设置第一列为索引
                except Exception as e:
                    print(f"⚠️ 无法解析 sheet: {sheet_name} in file {filename}，跳过该 sheet。原因：{e}")
                    continue

                if sheet_name not in sheet_data:
                    sheet_data[sheet_name] = [df]
                else:
                    sheet_data[sheet_name].append(df)

    # 合并所有 sheet，并按时间排序
    merged_sheets = {}
    for sheet_name, df_list in sheet_data.items():
        merged_df = pd.concat(df_list)
        merged_df = merged_df.sort_index()
        merged_sheets[sheet_name] = merged_df

    # 写入合并后的新 Excel 文件
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        for sheet_name, df in merged_sheets.items():
            df.to_excel(writer, sheet_name=sheet_name)

    print(f"✅ 所有 Excel 文件已成功合并，保存至：{output_path}")


def check_index(file_path, file_name):
    # 你的 Excel 文件路径  file_path
    data_path = os.path.join(file_path, file_name + '.xlsx')
    # 读取 Excel，假设第2行为列名，第一列为时间列
    df = pd.read_excel(data_path, header=1)
    df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0], errors='coerce')  # 将第一列转为 datetime
    df = df.dropna(subset=[df.columns[0]])  # 删除无效时间
    df = df.set_index(df.columns[0])

    # 保证 index 类型正确并按时间排序
    df = df.sort_index()

    # 生成完整时间序列（15分钟间隔）
    start_time = df.index.min()
    end_time = df.index.max()
    full_range = pd.date_range(start=start_time, end=end_time, freq='15T')

    # 检查缺失和重复
    missing_times = full_range.difference(df.index)
    duplicate_times = df.index[df.index.duplicated()]

    # 输出结果
    print("📌 总数据时间范围：", start_time, " ~ ", end_time)
    print("🔍 缺失的时间戳数量：", len(missing_times))
    if len(missing_times) > 0:
        print("缺失时间戳列表：")
        print(missing_times)

    print("🔍 重复的时间戳数量：", len(duplicate_times))
    if len(duplicate_times) > 0:
        print("重复时间戳列表：")
        print(duplicate_times)


def clean_index(input_file, file_name):
    # 文件路径设置
    data_path = os.path.join(file_path, file_name + '.xlsx')
    output_file = os.path.join(input_file,'{}_cleaned.xlsx'.format(file_name))  # 输出文件
    # 读取 Excel 中的所有 sheet
    xls = pd.ExcelFile(data_path, engine='openpyxl')
    sheet_names = xls.sheet_names

    # 创建写入器
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        for sheet in sheet_names:
            print(f"🛠 正在处理 sheet：{sheet}")
            try:
                # 读取当前 sheet，假设第一行为列名，第一列为时间列
                df = pd.read_excel(data_path, sheet_name=sheet, engine='openpyxl')
                df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0], errors='coerce')
                df = df.dropna(subset=[df.columns[0]])  # 删除无效时间
                df = df.set_index(df.columns[0])
                df = df.sort_index()

                # 删除重复时间索引，保留第一次出现的
                df = df[~df.index.duplicated(keep='first')]

                # 生成完整时间范围
                full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='15T')

                # 补全缺失时间索引
                df = df.reindex(full_range)

                # 线性插值：仅当前后都有值时才会填补，其他保持 NaN
                df = df.interpolate(method='time', limit_direction='both', limit_area='inside')

                # 将 index 恢复为列（写入 Excel 用）
                df.reset_index(inplace=True)
                df.rename(columns={'index': 'date'}, inplace=True)  # 可根据实际中文名称修改

                # 写入处理后的 sheet
                df.to_excel(writer, sheet_name=sheet, index=False)

            except Exception as e:
                print(f"❌ 处理 sheet {sheet} 时出错，跳过。错误信息：{e}")

    print(f"✅ 所有 sheet 处理完成，结果已保存至：{output_file}")


if __name__ == '__main__':
    # merge_data()

    file_path = r'D:\Time-LLM-main\dataset\HVAC'
    file_name = 'chilled_water_pump'
    check_index(file_path,file_name)
    clean_index(file_path,file_name)
    #
    file_name = '{}_cleaned'.format(file_name)
    check_index(file_path,file_name)

    # file_path = r'D:\Time-LLM-main\dataset\HVAC\cooling_tower.xlsx'
    # check_index(file_path)

