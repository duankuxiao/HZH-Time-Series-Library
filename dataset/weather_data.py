import os
import urllib.request  # pip install urllib

import pandas as pd
from bs4 import BeautifulSoup  # pip3 install Beautifulsoup4
import re
import datetime

def str2float(data):
    try:
        return float(data)
    except:
        return 0

def JMA_scrap_1day(date,city='Tokyo'):
    '''

    :param date: 日付
    :return: 1日ごと気象庁気象データ
    '''

    if city == 'Tokyo':
        url = "http://www.data.jma.go.jp/obd/stats/etrn/view/hourly_s1.php?prec_no=44&block_no=47662&year=%d&month=%d&day=%d&view=" % (date.year, date.month, date.day)  # 東京（東京都)
    elif city == 'Sapporo':
        url = "http://www.data.jma.go.jp/obd/stats/etrn/view/hourly_s1.php?prec_no=14&block_no=47412&year=%d&month=%d&day=%d&view=" % (date.year, date.month, date.day)  # 札幌（北海道)
    elif city == 'Naha':
        url = "http://www.data.jma.go.jp/obd/stats/etrn/view/hourly_s1.php?prec_no=91&block_no=47936&year=%d&month=%d&day=%d&view=" % (date.year, date.month, date.day)  # 那霸（冲绳）
    elif city == 'Sendai':
        url = "http://www.data.jma.go.jp/obd/stats/etrn/view/hourly_s1.php?prec_no=34&block_no=47590&year=%d&month=%d&day=%d&view=" % (date.year, date.month, date.day)  # 仙台（宫城）
    elif city == 'Osaka':
        url = "http://www.data.jma.go.jp/obd/stats/etrn/view/hourly_s1.php?prec_no=62&block_no=47772&year=%d&month=%d&day=%d&view=" % (date.year, date.month, date.day)  # 大阪（大阪）
    elif city == 'Fukuoka':
        url = "http://www.data.jma.go.jp/obd/stats/etrn/view/hourly_s1.php?prec_no=82&block_no=47807&year=%d&month=%d&day=%d&view=" % (date.year, date.month, date.day)  # 福冈（福冈）
    else:
        raise ValueError('Unknown city')

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


def get_JMA_data(start_date, end_date, city='Tokyo', output=True):
    '''

    :param startdate: 開始日付
    :param enddate: 終了日付
    :param output: csv出力
    :return: 気象庁気象データをいれたDataFrame
    '''
    if isinstance(start_date, str):
        start_date = datetime.datetime.strptime(start_date, '%Y/%m/%d')
    if isinstance(end_date, str):
        end_date = datetime.datetime.strptime(end_date, '%Y/%m/%d')
    date = start_date
    meteorologicaldata = pd.DataFrame(
        columns=['Year', 'Month', 'Day', 'Hour', 'Atmospheric_pressure_local', 'Atmospheric_pressure_sea_level', 'Precipitation', 'Temperature', 'Dew_point', 'Vapor_pressure', 'Relative_humidity',
                 'Wind_speed', 'Wind_direction', 'Sunshine_duration', 'Global_horizontal_irradiance', 'Snowfall', 'Snow_accumulation', 'Weather', 'Cloud_cover'])
    # ["年","月","日", "時間", "気圧（現地）", "気圧（海面）","降水量", "気温", "露点湿度", "蒸気圧", "湿度", "風速", "風向", "日照時間", "全天日射量", "降雪", "積雪","天気","雲量"]
    index_list = []
    day_count = 0
    try:
        while date != end_date + datetime.timedelta(days=1):

            data_per_hour_list = JMA_scrap_1day(date,city=city)
            for data_per_hour in data_per_hour_list:
                index_date = datetime.datetime(date.year, date.month, date.day, 1, 0, 0)
                index = data_per_hour_list.index(data_per_hour)
                index_date = index_date + datetime.timedelta(hours=index)
                index_date = datetime.datetime(index_date.year, index_date.month, index_date.day, index_date.hour, 0, 0)
                index_list.append(index_date)
                index = index + day_count * 24
                meteorologicaldata.loc[index, 'Year'] = index_date.year
                meteorologicaldata.loc[index, 'Month'] = index_date.month
                meteorologicaldata.loc[index, 'Day'] = index_date.day
                meteorologicaldata.loc[index, 'Hour'] = index_date.hour
                for i in range(len(data_per_hour) - 1):
                    try:
                        data = str2float(data_per_hour[i][0])
                        meteorologicaldata.iloc[index, i + 4] = data
                    except:
                        meteorologicaldata.iloc[index, i + 4] = '--'
            day_count += 1
            date += datetime.timedelta(days=1)

            startdate_str = start_date.strftime('%Y%m%d')
            enddate_str = end_date.strftime('%Y%m%d')
            meteorologicaldata.index = index_list
            if output == True:
                meteorologicaldata.to_csv(
                    r'D:\Time-LLM-main\dataset\solar_radiation\{}_{}_{}.csv'.format(city,startdate_str, enddate_str))
        return meteorologicaldata

    except:
        print('{}-{}-{} Meteorological data are not updated'.format(start_date.year, start_date.month, start_date.day))


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


if __name__ == '__main__':
    start_date = '2020/1/1'
    end_date = '2023/12/31'
    city = 'Sapporo'  # Sapporo Sendai Tokyo Osaka Fukuoka Naha

    jma_data = get_JMA_data(start_date, end_date, city=city)


