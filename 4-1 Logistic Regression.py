#%%
# csv 데이터 인터넷에서 불러오기
import pandas as pd
fish = pd.read_csv('https://bit.ly/fish_csv_data')
fish.head()
# >
#                Species	Weight	Length	Diagonal    Height	Width
#               0  Bream	242.0	 25.4	  30.0	   11.5200	4.0200
#               1	Bream	290.0  	 26.3	  31.2	   12.4800	4.3056
#               2	Bream	340.0	 26.5	  31.1	   12.3778	4.6961
#               3	Bream	363.0	 29.0	  33.5	   12.7300	4.4555
#               4	Bream	430.0	 29.0	  34.0	   12.4440	5.1340

#%%
# 'Species' 열에 있는 고유값 출력
print(pd.unique(fish['Species']))

# >                 ['Bream' 'Roach' 'Whitefish' 'Parkki' 'Perch' 'Pike' 'Smelt']
#%%
# 'Species' 열을 빼고 나머지 데이터 fish_input에 저장, 하는김에 numpy 배열로 바꿈.
fish_input = fish[['Weight', 'Length', 'Diagonal', 'Height', 'Width']].to_numpy()

print(fish_input[:5])
#>
#                        [[242.      25.4     30.      11.52     4.02  ]
#                         [290.      26.3     31.2     12.48     4.3056]
#                         [340.      26.5     31.1     12.3778   4.6961]
#                         [363.      29.      33.5     12.73     4.4555]
#                         [430.      29.      34.      12.444    5.134 ]]

#%%

fish_target = fish['Species'].to_numpy()
print(fish_target[:5])

#>                           ['Bream' 'Bream' 'Bream' 'Bream' 'Bream']

#%%
# 머신러닝 데이터세트 만들기

from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(fish_input, fish_target, random_state=42)

# StandardScaler 표준화 전처리.
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)


#%%
# K-최근접 이웃 분류기의 확률 예측

from sklearn.neighbors import KNeighborsClassifier
kn = KNeighborsClassifier(n_neighbors=3)
kn.fit(train_scaled, train_target)
print(kn.score(train_scaled, train_target))
print(kn.score(test_scaled, test_target))

#>                              0.8907563025210085
#                               0.85

#%%
print(kn.classes_)

#>                          ['Bream' 'Parkki' 'Perch' 'Pike' 'Roach' 'Smelt' 'Whitefish']

print(test_scaled[:5])

#>                          [[-0.88741352 -0.91804565 -1.03098914 -0.90464451 -0.80762518]
#                            [-1.06924656 -1.50842035 -1.54345461 -1.58849582 -1.93803151]
#                            [-0.54401367  0.35641402  0.30663259 -0.8135697  -0.65388895]
#                            [-0.34698097 -0.23396068 -0.22320459 -0.11905019 -0.12233464]
#                            [-0.68475132 -0.51509149 -0.58801052 -0.8998784  -0.50124996]]

print(kn.predict(test_scaled[:5]))

#>                          ['Perch' 'Smelt' 'Pike' 'Perch' 'Perch']

#%%
# 이 다섯개의 샘플에 대한 확률을 출력. (근처 세 마리에 대한 확률.)
# predict_proba()
# round() = 반올림.
# decimals = *int == 소수점 int자리까지 표기함.(반올림)
import numpy as np
proba = kn.predict_proba(test_scaled[:5])
print(np.round(proba, decimals=4))
#>
#                            [[0.     0.     1.     0.     0.     0.     0.    ]
#                            [0.     0.     0.     0.     0.     1.     0.    ]
#                            [0.     0.     0.     1.     0.     0.     0.    ]
#                            [0.     0.     0.6667 0.     0.3333 0.     0.    ]
#                            [0.     0.     0.6667 0.     0.3333 0.     0.    ]]

#  순서대로 첫번째 열부터 : ['Bream' 'Parkki' 'Perch' 'Pike' 'Roach' 'Smelt' 'Whitefish']
#      에 대한 확률.
#%%
distances, indexes = kn.kneighbors(test_scaled[3:4])
print(train_target[indexes])

#>                                      [['Roach' 'Perch' 'Perch']]

# test_scaled[3:4]의 주변 3개의 데이터. 
#%%
# Logistic Regression (로지스틱 회귀)
# z = a * (Weight) + b * (Length) + c * (Diagonal) + d * (Height) + e * (Width) + f
# z 는 어떤 값도 된다 -무한~ +무한. 하지만 확률로 변화하려면 0 ~ 1사이의 숫자여야하는데 여기서 쓰이는 ..

# -> 시그모이드 함수 (Sigmoid Function) or 로지스틱 함수 (Logistic Function)
#
#                       y = 1/(1+ e^(-z))       

import numpy as np
import matplotlib.pyplot as plt
z = np.arange(-5, 5, 0.1)
phi = 1 / (1+ np.exp(-z))
plt.plot(z, phi)
plt.xlabel('z')
plt.ylabel('phi')
plt.show()
#%%
# 로지스틱 회귀로 이진 분류 수행하기
# Boolean indexing.
# ex)
char_arr = np.array(['A', 'B', 'C', 'D', 'E'])
print(char_arr[[True, False, True, False, False]])

#>
#                                       ['A' 'C']
#%%
# 같은 방식으로 Bream 과 Smelt의 행만 골라낸 결과.
bream_smelt_indexes = (train_target == 'Bream') | (train_target == 'Smelt')
train_bream_smelt = train_scaled[bream_smelt_indexes]
target_bream_smelt = train_target[bream_smelt_indexes]

#%%
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(train_bream_smelt, target_bream_smelt)
print(lr.predict(train_bream_smelt[:5]))

#>                                      ['Bream' 'Smelt' 'Bream' 'Bream' 'Bream']
# 두번째 샘플을 제외하고 모두 도미로 예측함. 

print(lr.predict_proba(train_bream_smelt[:5]))

#>

#                                        [[0.99759855 0.00240145]
#                                        [0.02735183 0.97264817]
#                                        [0.99486072 0.00513928]
#                                        [0.98584202 0.01415798]
#                                        [0.99767269 0.00232731]]

print(lr.classes_)

#>                                          ['Bream' 'Smelt']


#%%
print(lr.coef_, lr.intercept_)

decisions = lr.decision_function(train_bream_smelt[:5])
print(decisions)
# decision_function() 로 z값 출력.(LogisticFunction)

#그리고 시그모디으 함수. expit()

from scipy.special import expit
print(expit(decisions))

#%%
# LogisticRegression 으로 다중 분류 수행.

lr = LogisticRegression(C = 20, max_iter=1000)
lr.fit(train_scaled, train_target)
print(lr.score(train_scaled, train_target))

print(lr.score(test_scaled, test_target))

#%%
print(lr.predict(test_scaled[:5]))

#%%
proba = lr.predict_proba(test_scaled[:5])
print(np.round(proba, decimals=3))

#%%
print(lr.classes_)
#%%
print(lr.coef_.shape, lr.intercept_.shape)
#%%
# 다중분류는 다르게 Softmax 함수를 사용하여 7개의 z 값을 확률로 변환한다.
decision = lr.decision_function(test_scaled[:5])
print(np.round(decision, decimals =2))







#%%
from scipy.special import softmax
proba = softmax(decision, axis=1)
print(np.round(proba, decimals=3))
