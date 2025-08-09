import os
import urllib.request  # pip install urllib

import numpy as np
from tqdm import tqdm
import pandas as pd
from bs4 import BeautifulSoup  # pip3 install Beautifulsoup4
import re
import datetime


def str2float(data):
    try:
        return float(data)
    except:
        return data


def JMA_scrap_1day(date,city = 'Tokyo',freq = 'hour'):
    '''

    :param date: 日付
    :return: 1日ごと気象庁気象データ
    '''

    if city == 'Tokyo':
        prec_no, block_no = 44,47662
    elif city == 'Sapporo':
        prec_no, block_no = 14,47412
    elif city == 'Naha':
        prec_no, block_no = 91,47936
    elif city == 'Sendai':
        prec_no, block_no = 34,47590
    elif city == 'Osaka':
        prec_no, block_no = 62,47772
    elif city == 'Fukuoka':
        prec_no, block_no = 82,47807
    elif city == 'Chiba':
        prec_no, block_no = 45,47682
    if freq == 'hour':
        url = "http://www.data.jma.go.jp/obd/stats/etrn/view/hourly_s1.php?prec_no=%d&block_no=%d&year=%d&month=%d&day=%d&view=" % (prec_no,block_no,date.year, date.month, date.day)  # 東京（東京都)
    elif freq == '10min':
        url = "https://www.data.jma.go.jp/stats/etrn/view/10min_s1.php?prec_no=%d&block_no=%d&year=%d&month=%d&day=%d&view=" % (prec_no,block_no,date.year, date.month, date.day)  # 東京（東京都)


    html = urllib.request.urlopen(url).read()
    soup = BeautifulSoup(html, 'html.parser')
    tr_list = soup.find("table", {"class": "data2_s"})
    data_per_hour_list = []
    # table の中身を取得
    for tr in tr_list.findAll('tr',{'class':'mtx'})[2:]:
        td_list = tr.findAll('td',{'class':'data_0_0'})
        p = r'<td .*?>(.*?)</td>'
        data_list = []
        for td in td_list:
            data = re.findall(p,str(td))
            if td_list.index(td) == 13:
                p_weather = r'<img alt="(.*?)" src=.*?>'
                data = re.findall(p_weather,str(td))
            data_list.append(data)
        data_per_hour_list.append(data_list)

    return data_per_hour_list


def TWH_scraping_48h():
    '''

    :return: The Weather Channel気象データ
    '''
    url = 'https://weather.com/ja-JP/weather/hourbyhour/l/4e73be5bb1986d9da1aea69b87c0c9992ad65aecd6fa642d1307f8c488fb6803'  # 東京都
    # url = 'https://weather.com/ja-JP/weather/hourbyhour/l/1321906f6eaa3be6e74276d04b253cbe737ac8dc30f3f99bbe784fa4471e27e8' # 浦安市
    html = urllib.request.urlopen(url)
    soup = BeautifulSoup(html.read(), 'html.parser')
    temp = soup.find_all("span", {"data-testid": "TemperatureValue",
                                  "class": "DetailsSummary--tempValue--jEiXE"})  # Temperature
    PoP_RH_C = soup.find_all("span",
                             {"data-testid": "PercentageValue"})  # Relative Humidity, Probability of Precipitation, Degree of Cloudiness
    Acc = soup.find_all("span", {"data-testid": "AccumulationValue", "class": "DetailsTable--value--2YD0-"}) # Precipitation
    Weather = soup.find_all("span", {"class": "DetailsSummary--extendedData--307Ax"}) # Weather
    W = soup.find_all("span", {"data-testid": "Wind","class": "Wind--windWrapper--3Ly7c DetailsTable--value--2YD0-"})

    p_temp = r'<span .*?>(.*?)<span>°</span>'  # r'<span .*?>(.*?)<span>°</span>'  # r'<span .*?>(.*?)°</span>'
    p_PoP_RH_C = r'<span .*?>(.*?)%</span>'
    p_Acc = r'<span .*?><span>(.*?)</span>'  #  r'<span .*?><span>(.*?)</span>' # r'<span .*?>(.*?)</span>'
    p_Weather = r'<span .*?>(.*?)</span>'
    p_w = r'<span .*?><span>(.*?) </span><span>(.*?)</span>\xa0<span .*?></span>' #r'<span .*?>(.*?) <!-- -->(.*?)</span>'

    temp_data = re.findall(p_temp, str(temp))
    PoP_RH_C_data = re.findall(p_PoP_RH_C, str(PoP_RH_C))
    Acc_data = re.findall(p_Acc, str(Acc))
    Weather_data = re.findall(p_Weather, str(Weather))
    W_data = re.findall(p_w,str(W))

    PoP_data = []
    RH_data = []
    C_data = []
    WS_data = []
    WD_data = []
    for i in range(len(PoP_RH_C_data)):
        if i % 3 == 0:
            PoP_data.append(PoP_RH_C_data[i])
        if i % 3 == 1:
            RH_data.append(PoP_RH_C_data[i])
        if i % 3 == 2:
            C_data.append(PoP_RH_C_data[i])
    for i in W_data:
        WS_data.append(i[1][:-5])
        WD_data.append((i[0]))
    return temp_data, PoP_data, RH_data, C_data, Acc_data, Weather_data,WS_data,WD_data


def get_JMA_data(start_date, end_date, city='Tokyo', freq='hour', output=True):
    '''
    获取气象厅气象数据

    :param start_date: 开始日期（字符串或 datetime）
    :param end_date: 结束日期（字符串或 datetime）
    :param city: 城市名称
    :param freq: 'hour' 或 '10min'
    :param output: 是否保存为 CSV
    :return: 包含气象数据的 DataFrame
    '''
    if isinstance(start_date, str):
        start_date = datetime.datetime.strptime(start_date, '%Y/%m/%d')
    if isinstance(end_date, str):
        end_date = datetime.datetime.strptime(end_date, '%Y/%m/%d')

    date = start_date

    if freq == 'hour':
        columns = ['Year', 'Month', 'Day', 'Hour', 'Atmospheric_pressure_local', 'Atmospheric_pressure_sea_level',
                   'Precipitation', 'Temperature', 'Dew_point', 'Vapor_pressure', 'Relative_humidity',
                   'Wind_speed', 'Wind_direction', 'Sunshine_duration', 'Global_horizontal_irradiance',
                   'Snowfall', 'Snow_accumulation', 'Weather', 'Cloud_cover']
    elif freq == '10min':
        columns = ['Year', 'Month', 'Day', 'Hour', 'Min', 'Atmospheric_pressure_local', 'Atmospheric_pressure_sea_level',
                   'Precipitation', 'Temperature', 'Relative_humidity', 'Wind_speed_mean', 'Wind_direction_mean',
                   'Wind_speed_max', 'Wind_direction_max', 'Sunshine_duration']
    else:
        raise ValueError("Only 'hour' and '10min' frequencies are supported.")

    data_frames = []

    try:
        while date <= end_date:
            try:
                data_list = JMA_scrap_1day(date, city=city, freq=freq)
                daily_df = pd.DataFrame(columns=columns)
                index_list = []

                if freq == 'hour':
                    for index in range(len(data_list)):
                        data_per_hour = data_list[index]
                        index_date = datetime.datetime(date.year, date.month, date.day, 1) + datetime.timedelta(hours=index)
                        index_list.append(index_date)

                        row = [index_date.year, index_date.month, index_date.day, index_date.hour]
                        for i in range(len(data_per_hour) - 1):
                            try:
                                row.append(str2float(data_per_hour[i][0]))
                            except:
                                row.append(None)
                        daily_df.loc[len(daily_df)] = row

                elif freq == '10min':
                    for index in range(len(data_list)):
                        data_per_10min = data_list[index]
                        index_date = datetime.datetime(date.year, date.month, date.day) + datetime.timedelta(minutes=10 * (index + 1))
                        index_list.append(index_date)

                        row = [index_date.year, index_date.month, index_date.day, index_date.hour, index_date.minute]
                        for i in range(len(data_per_10min)):
                            try:
                                row.append(str2float(data_per_10min[i][0]))
                            except:
                                row.append(None)
                        daily_df.loc[len(daily_df)] = row

                daily_df.index = index_list
                data_frames.append(daily_df)
                print(f"✅ {date.strftime('%Y-%m-%d')} done.")

            except Exception as e:
                print(f"❌ Error on {date.strftime('%Y-%m-%d')}: {e}")

            date += datetime.timedelta(days=1)

        # 合并所有天的数据
        meteorologicaldata = pd.concat(data_frames)
        meteorologicaldata.index.name = 'Datetime'
        meteorologicaldata['Precipitation'] = pd.to_numeric(meteorologicaldata['Precipitation'], errors='coerce')
        meteorologicaldata['Sunshine_duration'] = pd.to_numeric(meteorologicaldata['Sunshine_duration'], errors='coerce')
        meteorologicaldata['Global_horizontal_irradiance'] = pd.to_numeric(meteorologicaldata['Global_horizontal_irradiance'], errors='coerce')

        meteorologicaldata['Precipitation'] = meteorologicaldata['Precipitation'].fillna(0)
        meteorologicaldata['Sunshine_duration'] = meteorologicaldata['Sunshine_duration'].fillna(0)
        meteorologicaldata['Global_horizontal_irradiance'] = meteorologicaldata['Global_horizontal_irradiance'].fillna(0)

        # 输出为 CSV 文件
        if output:
            startdate_str = start_date.strftime('%Y%m%d')
            enddate_str = end_date.strftime('%Y%m%d')
            filename = f'{city}_{startdate_str}_{enddate_str}_{freq}.csv'
            meteorologicaldata.to_csv(filename, encoding='SHIFT-JIS')
            print(f"✅ 文件已保存为：{filename}")

        return meteorologicaldata

    except Exception as e:
        print("发生异常:", e)
        return pd.DataFrame()


def get_TWH_data(output=True):
    '''

    :param output: csv出力
    :return: The Weather Channel気象データをいれたDataFrame
    '''
    meteorologicaldata = pd.DataFrame(
        columns=['Year', 'Month', 'Day', 'Hour', 'Temp', 'RH', 'WS', 'WD', 'PoP', 'C', 'ACC', 'climate'])
    temp_data, PoP_data, RH_data, C_data, Acc_data, Weather_data, WS_data, WD_data = TWH_scraping_48h()
    currenttime = datetime.datetime.now()
    index_list = []
    for i in range(24):
        nexttime = currenttime + datetime.timedelta(hours=i + 1)
        index_date = datetime.datetime(nexttime.year, nexttime.month, nexttime.day, nexttime.hour, 0, 0)
        index_list.append(index_date)
        meteorologicaldata.loc[i, 'Year'] = nexttime.year
        meteorologicaldata.loc[i, 'Month'] = nexttime.month
        meteorologicaldata.loc[i, 'Day'] = nexttime.day
        meteorologicaldata.loc[i, 'Hour'] = nexttime.hour
        meteorologicaldata.loc[i, 'Temp'] = temp_data[i]
        meteorologicaldata.loc[i, 'RH'] = RH_data[i]
        meteorologicaldata.loc[i, 'WS'] = str2float(WS_data[i])
        meteorologicaldata.loc[i, 'WD'] = WD_data[i]
        meteorologicaldata.loc[i, 'PoP'] = PoP_data[i]
        meteorologicaldata.loc[i, 'C'] = C_data[i]
        meteorologicaldata.loc[i, 'ACC'] = Acc_data[i]
        meteorologicaldata.loc[i, 'climate'] = Weather_data[i]

    # print(weather_data)
    currenttime_str = currenttime.strftime('%Y%m%d%H')
    endtime = currenttime + datetime.timedelta(hours=48)
    endtime_str = endtime.strftime('%Y%m%d%H')
    meteorologicaldata.index = index_list
    if output == True:
        meteorologicaldata_file = r'D:\発電量予測\pycode\meteorological_data\TWH_data_{}_{}.csv'.format(currenttime_str,
                                                                                                        endtime_str)
        meteorologicaldata.to_csv(meteorologicaldata_file)
    return meteorologicaldata


def get_PV_data():
    '''
    :return: 発電量実測データをいれたDataFrame
    '''
    filepath = r"D:\発電量予測\pycode\measured_data"
    filename_measureddata = '実測データ.csv'
    data = pd.read_csv(os.path.join(filepath, filename_measureddata), index_col=0)
    data.index = pd.to_datetime(data.index, format='%Y/%m/%d %H:%M:%S')
    return data


def get_data(start_date, output=True):
    '''

    :param start_date:前日の日付
    :param output: 入力データをCSVに出力
    :return: AIモデルに入力するデータをいれたDataFrame
    '''
    end_date = start_date + datetime.timedelta(days=1)

    JMA_data = get_JMA_data(start_date, end_date)
    TWH_data = get_TWH_data()
    data_measured = get_PV_data()

    data_total = pd.concat([JMA_data, data_measured], axis=1, join='inner')
    data_total = pd.concat([data_total, TWH_data])

    data_total.index.name = 'Time'

    data = pd.DataFrame(columns=['Month', 'Day', 'Hour'], index=data_total.index)

    data['Month'] = data.index.month
    data['Day'] = data.index.day
    data['Hour'] = data.index.hour
    data['Temp'] = data_total['Temp']
    data['RH'] = data_total['RH']
    data['PV_power'] = data_total['発電_実測']
    data = data.interpolate(limit=50)
    if output == True:
        data.to_csv(r'./data.csv')
    return data


import requests
import pandas as pd
import numpy as np
from tqdm import tqdm
import time, json, os


def spider(req_url, sleep_time, start_date, end_date):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.127 Safari/537.36",
        "X-Requested-With": "XMLHttpRequest"
    }
    records_list = []
    date_list = pd.date_range(start_date, end_date, freq='D').strftime('%Y%m%d')

    for i in tqdm(date_list):
        url = req_url + '&startDate=' + i + '&endDate=' + i
        try:
            respon = requests.get(url, timeout=30, headers=headers)
            if respon.status_code == 200:
                data = json.loads(respon.text)['observations']
                records_list = records_list + data
            else:
                with open('error.txt', 'a+') as f0:
                    f0.write(i + ',' + respon.status_code + ',' + '\n')
        except Exception as e:
            print(e)
            with open('error.txt', 'a+') as f1:
                f1.write(i + '\n')
        time.sleep(sleep_time)

    df = pd.DataFrame.from_records(records_list)
    df.to_csv('data.csv', encoding='utf-8', index=False)
    return None


# 读取error.txt文件的内容，对错误的参数重新爬取
def supply(req_url, sleep_time):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.127 Safari/537.36",
        "X-Requested-With": "XMLHttpRequest"
    }
    records_list = []
    with open('error.txt', 'r') as f:
        lines = f.readlines()
        date_list = list(map(lambda x: x[0:8], lines))

    for i in tqdm(date_list):
        url = req_url + '&startDate=' + i + '&endDate=' + i
        try:
            respon = requests.get(url, timeout=30, headers=headers)
            if respon.status_code == 200:
                data = json.loads(respon.text)['observations']
                records_list = records_list + data
            else:
                with open('error.txt', 'a+') as f0:
                    f0.write(i + ',' + respon.status_code + ',' + '\n')
        except Exception as e:
            print(e)
            with open('error.txt', 'a+') as f1:
                f1.write(i + '\n')
        time.sleep(sleep_time)

    df = pd.DataFrame.from_records(records_list)
    df.to_csv('append_data.csv', encoding='utf-8', index=False)
    return None


def concat(file0, file1):
    df0 = pd.read_csv(file0, encoding='utf-8')
    df1 = pd.read_csv(file1, encoding='utf-8')
    df2 = pd.concat([df0, df1])
    return df2


def cleanData(df0, reset_start_time_index, reset_end_time_index):
    '''
    day_ind: 白天还是晚上
    temp: 气温
    wx_icon: 不知道用来干什么，其日变化不明显，只有三种值
    icon_extd: wx_icon后面补两个0，也是不知道用来干什么
    wx_phrase: 文字描述天气状况，如晴朗（fair），cloudy（多云）， windy（有风）
    dewPt: 露点温度
    heat_index:
    pressure: 气压
    vis: 可视范围
    wc: 未知
    wdir: 风向 0-365度 【注意这个理应被当成一个类型变量！！！！】
    wdir_cardinal: 风基数 文字描述
    gust: 阵风，不知道用来干什么，而且只有4819条记录，扔掉
    wspd: 风速
    uv_desc: 紫外线强度 文字描述
    feels_like: 体感温度 （作用不大）
    uv_index: 紫外线指数 整数描述
    clds: 未知
    humidity: 相对湿度
    suntime: 日照时数，根据 uv index 来计算，uv index为0就是黑夜，正就是白天，其中1是阴雨天。
    '''

    assert isinstance(df0, pd.DataFrame)

    # 注意! 由于爬下来的时间是GMT时间，所以使用apply函数对 “valid_time_gmt” 这一列数据增加 8h ，即28800秒，这样得到的就是北京时间
    # 如果需要爬取其他时区的数据，需要额外调整
    df0['date'] = pd.to_datetime(df0['valid_time_gmt'].apply(lambda x: x + 28800), unit='s')

    # 去掉无用字段
    df1 = df0.drop(['key', 'class', 'obs_id', 'obs_name',
                    'valid_time_gmt', 'expire_time_gmt', 'day_ind',
                    'feels_like', 'wx_icon', 'icon_extd', 'gust'],
                   axis=1) \
        .set_index(['time']) \
        .sort_index() \
        # .dropna(how = 'all', axis = 1) # 有些列全为空值

    # 华氏度转摄氏度
    df1['temp'] = df1['temp'].apply(lambda x: (x - 32) / 1.8)  # 气温
    df1['dewPt'] = df1['dewPt'].apply(lambda x: (x - 32) / 1.8)  # 露点温度

    # 湿度转为浮点数，并重命名列名
    df1['humidity'] = df1['rh'] / 100
    df1 = df1.drop(['rh'], axis=1)

    # 重建时间索引，因部分时间索引丢失
    new_time_index = pd.date_range(start=reset_start_time_index,
                                   end=reset_end_time_index,
                                   freq='H')
    df1 = df1.reindex(new_time_index, fill_value=np.nan).sort_index()
    df1.index.name = 'time'

    '''
    # 以下将逐30min数据计算为日均数据
    # df1 = df1.reset_index()
    # df1['date'] = df1['time'].dt.date
    # df1 = df1.drop(['time'], axis = 1)

    # 根据紫外线指数从大于0的时间计算当天日照时数
    # 去掉 0 ，对剩余的进行计数，单位为半小时
    def calculateSuntime(x):
        if x == 0:
            return 0
        else:
            return 0.5
    df_suntime = df1[['date', 'uv_index']].copy()
    df_suntime['uv_index'] = df_suntime['uv_index'].apply(calculateSuntime)
    df_suntime = df_suntime.groupby(['date']).sum()
    df_suntime.rename(columns = {'uv_index':'suntime'}, inplace = True)


    # 计算日均、日最大、日最小 （仅针对64位浮点类型变量）
    df_float64 = df1[['date', 'temp', 'dewPt', 'heat_index', 'pressure', 'vis', 'wc', 'wspd', 'humidity']]
    df_float64_mean = df_float64.groupby(['date']).mean()
    df_float64_mean.rename(columns={'temp':'temp_mean',
                                    'dewPt':'dewPt_mean',
                                    'heat_index':'heat_index_mean',
                                    'pressure':'pressure_mean',
                                    'vis':'vis_mean',
                                    'wc':'wc_mean',
                                    'wspd':'wspd_mean',
                                    'humidity':'humidity_mean'},
                                    inplace = True)
    df_float64_max = df_float64.groupby(['date']).max()
    df_float64_max.rename(columns={'temp':'temp_max',
                                    'dewPt':'dewPt_max',
                                    'heat_index':'heat_index_max',
                                    'pressure':'pressure_max',
                                    'vis':'vis_max',
                                    'wc':'wc_max',
                                    'wspd':'wspd_max',
                                    'humidity':'humidity_max'},
                                    inplace = True)
    df_float64_min = df_float64.groupby(['date']).min()
    df_float64_min.rename(columns={'temp':'temp_min',
                                    'dewPt':'dewPt_min',
                                    'heat_index':'heat_index_min',
                                    'pressure':'pressure_min',
                                    'vis':'vis_min',
                                    'wc':'wc_min',
                                    'wspd':'wspd_min',
                                    'humidity':'humidity_min'},
                                    inplace = True)
    df_float64_1 = pd.concat([df_float64_mean, df_float64_min, df_float64_max, df_suntime], axis = 1, verify_integrity = True)


    # 类型变量，计算当日众数
    df_object = df1[['date', 'wx_phrase', 'wdir', 'uv_desc', 'wdir_cardinal', 'clds']]
    df_object_mode = df_object.groupby(['date']).agg(lambda x: x.value_counts().index[0])
    df_object_mode.rename(columns = {'wx_phrase':'wx_phrase_mode',
                                    'wdir':'wdir_mode',
                                    'uv_desc':'uv_desc_mode',
                                    'wdir_cardinal':'wdir_cardinal_mode',
                                    'clds':'clds_mode'},
                                    inplace = True)

    # 合并数据
    df2 = pd.concat([df_float64_1, df_object_mode], axis = 1, verify_integrity = True)
    '''

    # 得到的数据是逐半小时的，重置索引为逐小时
    # new_index = pd.date_range('12/29/2009', periods = 10, freq = 'D')
    # df1.reindex(new_index)

    df1.to_csv('Qingdao.csv', encoding='utf-8')
    os.remove('data.csv')

    return None


if __name__ == '__main__':
    # 中国天气数据爬虫
    req_url = 'https://api.weather.com/v1/location/ZSQD:9:CN/observations/historical.json?apiKey=e1f10a1e78da46f5b10a1e78da96f525&units=e'  # 青岛 ZSQD  广州白云机场 ZGGG
    start_date = '2023-03-06'
    end_date = '2024-08-20'  # '2024-08-20'
    sleep_time = 10  # 最好拉长睡眠时间，太短的时间（频繁请求）回被封掉ip地址，6秒以上差不多

    #########################################################################################################################################################
    spider(req_url=req_url, sleep_time=sleep_time, start_date=start_date, end_date=end_date)

    if os.path.isfile('error.txt'):
        supply(req_url=req_url, sleep_time=sleep_time)
        concat_df = concat('append_data_.csv', 'data.csv')
    else:
        concat_df = pd.read_csv('data.csv', encoding='utf-8')

    cleanData(
        concat_df,
        reset_start_time_index=start_date + ' 00:00:00',
        reset_end_time_index=end_date + ' 00:00:00'
    )

    # 日本天气数据爬虫
    # start_date = '2016/1/1'
    # end_date = '2019/12/31'
    # city = 'Tokyo'  # Sapporo Sendai Tokyo Osaka Fukuoka Naha Chiba
    # jma_data = get_JMA_data(start_date, end_date, freq='10min', city=city)
    # # jma_data = get_JMA_data(start_date, end_date, freq='hour', city=city)
    #
    # # start_date = '2024/12/6'
    # # end_date = '2025/2/24'
    # # city = 'Chiba'  # Sapporo Sendai Tokyo Osaka Fukuoka Naha Chiba
    # # jma_data = get_JMA_data(start_date, end_date, freq='10min', city=city)
    #
    # jma_data = pd.read_csv('Chiba_20241206_20250224_10min.csv',index_col=0,encoding='SHIFT-JIS')
    # jma_data.index = pd.to_datetime(jma_data.index)
    #
    # df_resampled = jma_data.resample('5T').interpolate(method='linear')
    # df_resampled.to_csv('Chiba_20241206_20250224_5min.csv',encoding='SHIFT-JIS')

    # data = pd.read_csv('Chiba_20241207_20250204_5min.csv',encoding='SHIFT-JIS',index_col=0)
    # print(data.columns)
