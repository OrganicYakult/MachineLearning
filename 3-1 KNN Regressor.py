#%%
import numpy as np
#%%
perch_length = np.array([8.4, 13.7, 15.0, 16.2, 17.4, 18.0, 18.7, 19.0, 19.6, 20.0, 21.0,
       21.0, 21.0, 21.3, 22.0, 22.0, 22.0, 22.0, 22.0, 22.5, 22.5, 22.7,
       23.0, 23.5, 24.0, 24.0, 24.6, 25.0, 25.6, 26.5, 27.3, 27.5, 27.5,
       27.5, 28.0, 28.7, 30.0, 32.8, 34.5, 35.0, 36.5, 36.0, 37.0, 37.0,
       39.0, 39.0, 39.0, 40.0, 40.0, 40.0, 40.0, 42.0, 43.0, 43.0, 43.5,
       44.0])
perch_weight = np.array([5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0, 110.0,
       115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0, 130.0,
       150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0, 197.0,
       218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0, 514.0,
       556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0, 820.0,
       850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0, 1000.0,
       1000.0])

#%%
import matplotlib.pyplot as plt
plt.scatter(perch_length, perch_weight)
plt.xlabel('length')
plt.ylabel('weight')
plt.show()
#%%
from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(
    perch_length, perch_weight, random_state=42

)

print(train_input.shape, train_target.shape)
print(test_input.shape, test_target.shape)

#%%
test_array = np.array([1,2,3,4])
test_array.shape
test_array
#%%
test_array = test_array.reshape(2, 2)
test_array.shape
#%%
test_array
#%%
train_input
train_target
#%%
train_input.shape
train_target.shape
#%%
train_input = train_input.reshape(-1,1)
test_input = test_input.reshape(-1,1)
print(train_input, test_input)
print(train_input.shape, test_input.shape)
#%%
# .shape , .reshape
# - (column, row)
#%%
# KNN Regressor. 이웃 회귀 알고리즘. classifier 와 매우 비슷함.
from sklearn.neighbors import KNeighborsRegressor
knr = KNeighborsRegressor()
knr.fit(train_input, train_target)
knr.score(test_input, test_target)
# 0.992809406101064
# 회귀는 1.0이 나오는건 불가능하다. 
# 예측값이 임의의 수이기 때문.

# R^2  = 결정계수
#%%
from sklearn.metrics import mean_absolute_error

# 테스트 세트에 대한 예측을 만듦
test_prediction = knr.predict(test_input)

# 평균 오차값 확인 = mean_absolute_error
mae = mean_absolute_error(test_target, test_prediction)
mae
#%%
print(knr.score(train_input, train_target))     # 0.9698823289099254
print(knr.score(test_input, test_target))       # 0.992809406101064

# ...? 훈련세트가 더 높은 점수가 나와야하는데 낮다.
# == 훈련세트를 제대로 학습을 못했다. == 과소적합
# (테스트세트의 데이터가 적음.)
# 반대로 훈련세트가 굉장히 좋은 점수, 테스트세트가 굉장히 나쁘다면 == 과대적합.
#%%
knr.n_neighbors = 3

knr.fit(train_input, train_target)
print(knr.score(train_input, train_target))

print(knr.score(test_input, test_target))

