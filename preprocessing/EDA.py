import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
from scipy.stats import norm
import seaborn as sns

font_path = "C:/Windows/Fonts/malgunbd.ttf"
font_name = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font_name)
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 부호

data = pd.read_csv('C:/Users/user/Downloads/통합 문서1.csv', encoding='utf-8')

# 1. 타겟 변수 확인 (Distribution of Target)
figure, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
figure.set_size_inches(14,6)
sns.distplot(data['평단가'], fit=norm, ax=ax1)
sns.distplot(np.log(data['평단가']+1), fit=norm, ax=ax2)
plt.show()

# 2. 변수간 상관관계 확인 (Feature Correlation)
corr=data.corr()
top_corr=data[corr.nlargest(40,'평단가')['평단가'].index].corr()
figure, ax1 = plt.subplots(nrows=1, ncols=1)
figure.set_size_inches(20,15)
sns.heatmap(top_corr, annot=True, ax=ax1)
plt.show()

# 대형마트_수 와 평단가 의 관계를 표시한 그래프
sns.regplot(data['대형마트_수'], data['평단가'])
plt.show()