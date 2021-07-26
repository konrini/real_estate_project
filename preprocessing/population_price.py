# from matplotlib import font_manager
#
# for font in font_manager.fontManager.ttflist:
#     print(font.name, font.fname)
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from matplotlib import font_manager, rc

font_path = "C:/Windows/Fonts/malgunbd.ttf"
font_name = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font_name)

plt.rcParams['axes.unicode_minus'] = False  # 마이너스 부호

people = pd.read_csv('C:/Users/user/Downloads/백현동 유입인구.csv')
# people = pd.read_csv('C:/Users/user/Downloads/분당구 유입인구.csv')
people['평가평'] = people['평당 가격 평균']

for i in range(len(people['평당 가격 평균'])):
    people['평가평'][i] = (people['평당 가격 평균'][i] - min(people['평당 가격 평균'])) / (max(people['평당 가격 평균']) - min(people['평당 가격 평균']))

fig = plt.figure(figsize=(8,8)) ## 캔버스 생성
fig.set_facecolor('white') ## 캔버스 색상 설정
ax = fig.add_subplot() ## 그림 뼈대(프레임) 생성

ax.plot(people['연도'], people['평가평'], label='평균 평당 가격')
# ax.plot(people['연도'], people['인구수'], label='백현동 총 인구 수')

ax.legend()

plt.show()
