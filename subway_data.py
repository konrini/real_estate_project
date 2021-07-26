import pandas as pd

df = pd.read_csv('판교푸르지오그랑블실거래가_110713-210713.csv', encoding = 'utf-8')

# 계약날짜 yy-mm-dd 형식으로 바꾸기
from datetime import datetime
df['계약날짜'] = df['계약년월']
for i in range(len(df['거래금액(만원)'])):
    df['계약날짜'][i] = str(df['계약년월'][i]) + str(df['계약일'][i]).zfill(2)
    df['계약날짜'][i] = datetime.strptime(str(df['계약날짜'][i]), "%Y%m%d").date()
    df['거래금액(만원)'][i] = int(df['거래금액(만원)'][i][:-4] + df['거래금액(만원)'][i][-3:])

# 평당가격 구하기
df['평당가격'] = df['거래금액(만원)']
for i in range(len(df['거래금액(만원)'])):
    df['평당가격'][i] = (df['거래금액(만원)'][i]/df['전용면적(㎡)'][i])*3.3

# 평당가격과 노선 개통일 기준으로 분류     
graph_all_2cha = df[(pd.to_datetime('2016-06-30') >= df['계약날짜']) & (df['계약날짜'] >= pd.to_datetime('2015-07-01'))]
graph_all_1cha = df[(pd.to_datetime('2012-12-31') >= df['계약날짜'])]
graph_all_gg = df[(pd.to_datetime('2017-02-28') >= df['계약날짜']) & (df['계약날짜'] >= pd.to_datetime('2016-03-01'))]

# 10년치 평당가격 그래프
plt.plot(df['계약날짜'], df['평당가격'])
plt.title('전체 10년치')
plt.xlabel('계약날짜')
plt.ylabel('평당가격')
plt.show()

import matplotlib.pyplot as plt
from datetime import datetime
from matplotlib import font_manager, rc

# 그래프 폰트 설정
font_path = "C:/Windows/Fonts/malgun.ttf"
font_name = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font_name)

# 노선별 1년치 평당가격 막대그래프
plt.bar(graph_all_1cha['계약날짜'], graph_all_1cha['평당가격'])
plt.title('신분당선 1차 개통 2011-10')
plt.xlabel('계약날짜')
plt.ylabel('평당가격')
plt.ylim([min(graph_all_1cha['평당가격'])-100,max(graph_all_1cha['평당가격'])+100])
plt.show()

plt.bar(graph_all_2cha['계약날짜'], graph_all_2cha['평당가격'])
plt.title('신분당선 2차 개통 2016-01')
plt.xlabel('계약날짜')
plt.ylabel('평당가격')
plt.ylim([min(graph_all_2cha['평당가격'])-100,max(graph_all_2cha['평당가격'])+100])
plt.show()

plt.bar(graph_all_gg['계약날짜'], graph_all_gg['평당가격'])
plt.title('경강선 개통 2016-09')
plt.xlabel('계약날짜')
plt.ylabel('평당가격')
plt.ylim([min(graph_all_gg['평당가격'])-100,max(graph_all_gg['평당가격'])+100])
plt.show()

# 면적별 class 분류
very_small = df[(df['전용면적(㎡)'] == 97.7136) | (df['전용면적(㎡)'] == 98.9860)][['계약날짜','거래금액(만원)','층','전용면적(㎡)','계약년월', '계약일']].reset_index(drop = True).sort_values(by = ['계약년월', '계약일'])
small = df[(df['전용면적(㎡)'] == 103.9649) | (df['전용면적(㎡)'] == 105.1308)][['계약날짜','거래금액(만원)','층','전용면적(㎡)', '계약년월', '계약일']].reset_index(drop = True).sort_values(by = ['계약년월', '계약일'])
middle = df[(df['전용면적(㎡)'] == 117.5193)][['계약날짜','거래금액(만원)','층','전용면적(㎡)', '계약년월', '계약일']].reset_index(drop = True).sort_values(by = ['계약년월', '계약일'])
large = df[(df['전용면적(㎡)'] == 139.7268) | (df['전용면적(㎡)'] == 265.5543)][['계약날짜','거래금액(만원)','층','전용면적(㎡)', '계약년월', '계약일']].reset_index(drop = True).sort_values(by = ['계약년월', '계약일'])

# 기울기 구하는 함수
def gradient(dataframe, columns = '거래금액(만원)'):
    lst = []
    for i in range(1,len(dataframe[columns])):
        lst.append((dataframe[columns][i] - dataframe[columns][i-1])/(len(dataframe[columns])))
    return lst

# 평수 마다 가격 변화량
very_small['Grad'] = pd.DataFrame(gradient(very_small))
small['Grad'] = pd.DataFrame(gradient(small))
middle['Grad'] = pd.DataFrame(gradient(middle))
large['Grad'] = pd.DataFrame(gradient(large))

# 평수마다 노선 개통일 기준 가격 그래프
graph_all_2cha = df[(pd.to_datetime('2016-06-30') >= df['계약날짜']) & (df['계약날짜'] >= pd.to_datetime('2015-07-01'))]
graph_all_1cha = df[(pd.to_datetime('2012-12-31') >= df['계약날짜'])]
graph_all_gg = df[(pd.to_datetime('2017-02-28') >= df['계약날짜']) & (df['계약날짜'] >= pd.to_datetime('2016-03-01'))]

graph_vs_2cha = very_small[(pd.to_datetime('2016-06-30') >= very_small['계약날짜']) & (very_small['계약날짜'] >= pd.to_datetime('2015-07-01'))]
graph_vs_1cha = very_small[(pd.to_datetime('2012-12-31') >= very_small['계약날짜'])]
graph_vs_gg = very_small[(pd.to_datetime('2017-02-28') >= very_small['계약날짜']) & (very_small['계약날짜'] >= pd.to_datetime('2016-03-01'))]

graph_s_2cha = small[(pd.to_datetime('2016-06-30') >= small['계약날짜']) & (small['계약날짜'] >= pd.to_datetime('2015-07-01'))]
graph_s_1cha = small[(pd.to_datetime('2012-12-31') >= small['계약날짜'])]
graph_s_gg = small[(pd.to_datetime('2017-02-28') >= small['계약날짜']) & (small['계약날짜'] >= pd.to_datetime('2016-03-01'))]

graph_m_2cha = middle[(pd.to_datetime('2016-06-30') >= middle['계약날짜']) & (middle['계약날짜'] >= pd.to_datetime('2015-07-01'))]
graph_m_1cha = middle[(pd.to_datetime('2012-12-31') >= middle['계약날짜'])]
graph_m_gg = middle[(pd.to_datetime('2017-02-28') >= middle['계약날짜']) & (middle['계약날짜'] >= pd.to_datetime('2016-03-01'))]

graph_l_2cha = large[(pd.to_datetime('2016-06-30') >= large['계약날짜']) & (large['계약날짜'] >= pd.to_datetime('2015-07-01'))]
graph_l_1cha = large[(pd.to_datetime('2012-12-31') >= large['계약날짜'])]
graph_l_gg = large[(pd.to_datetime('2017-02-28') >= large['계약날짜']) & (large['계약날짜'] >= pd.to_datetime('2016-03-01'))]