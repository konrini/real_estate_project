import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc

font_path = "C:/Windows/Fonts/malgunbd.ttf"
font_name = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font_name)

apart = pd.read_csv('C:/Users/user/Downloads/간단)백현동아파트실거래가_100726-210727.csv')

apart_name = list(set(apart['단지명']))
apart2 = apart.groupby(['단지명', '계약년도'], as_index=False).mean()

fig = plt.figure(figsize=(8,8)) ## 캔버스 생성
fig.set_facecolor('white') ## 캔버스 색상 설정
ax = fig.add_subplot() ## 그림 뼈대(프레임) 생성

for i in range(len(apart_name)):
    ax.plot(apart2[apart2['단지명'] == apart_name[i]]['계약년도'], apart2[apart2['단지명'] == apart_name[i]]['평당가'], label=apart_name[i])

ax.legend()
plt.title('연도별 백현동 아파트 평당가')
plt.show()
