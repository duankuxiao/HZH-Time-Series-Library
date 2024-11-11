import numpy as np
import pandas as pd


# 定义一个函数，使用指定的插值方法
def interpolate_and_compare(df, fraction=0.05, feature='PV', method='linear', order=None):
    df.index = pd.to_datetime(df.index)
    # 随机去掉5%的值
    missing_indices = df.sample(frac=fraction).index  # 5%的索引
    df_with_missing = df.copy()
    df_with_missing.loc[missing_indices, feature] = np.nan

    # 使用指定的插值方法补全缺失值
    df_interpolated = df_with_missing.copy()
    if method in ['polynomial', 'spline'] and order is not None:
        if method == 'spline':
            assert 1 <= order <= 5, "order should be more than 1 and less than 5"
        df_interpolated[feature] = df_interpolated[feature].interpolate(method=method, order=order)
    else:
        df_interpolated[feature] = df_interpolated[feature].interpolate(method=method)

    # 计算插值误差，只比较原始的缺失位置
    true_values = df.loc[missing_indices, feature]
    interpolated_values = df_interpolated.loc[missing_indices, feature]
    interpolation_error = np.abs(true_values - interpolated_values)

    # 计算平均绝对误差（MAE）
    mae = interpolation_error.mean()
    print(f"插值误差的平均绝对误差（MAE），使用方法 '{method}' 和 order={order}. MAE:", round(mae,4))

    # 保存真值和插值结果到 CSV 文件
    comparison_df = pd.DataFrame({
        'True Values': true_values,
        'Interpolated Values': interpolated_values,
        'Interpolation Error': interpolation_error
    })
    file_name = f'./results/interpolation_comparison_{method}_{order}.csv'
    comparison_df.to_csv(file_name, index_label='Index')
    print(f"真值和插值结果已保存为 '{file_name}' 文件")


if __name__ == '__main__':
    df = pd.read_csv('./dataset/PV/PV_hour.csv', index_col=0)

    # 示例：使用不同的插值方法
    interpolate_and_compare(df, method='linear')
    interpolate_and_compare(df, method='polynomial', order=2)
    interpolate_and_compare(df, method='polynomial', order=5)
    interpolate_and_compare(df, method='spline', order=2)
    interpolate_and_compare(df, method='spline', order=5)
    interpolate_and_compare(df, method='time')

