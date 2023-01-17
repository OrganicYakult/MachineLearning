#%%
fish_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0, 
                31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0, 
                35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0, 9.8, 
                10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
fish_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0, 
                500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0, 
                700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0, 6.7, 
                7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]
#%%
import numpy as np

#%%
#                     np.
#   column_stack()            concatenate()
# / / /      o o o          / / /       o o o  
#                               
#       / o                    / / / o o o 
#       / o             
#       / o
np.column_stack(([1,2,3],[4,5,6]))
#%%
fish_data = np.column_stack((fish_length, fish_weight))
fish_data
#%%
np.ones(5)
#%%
fish_target = np.concatenate((np.ones(35), np.zeros(14)))
#%%
from sklearn.model_selection import train_test_split

train_input, test_input, train_target, test_target = train_test_split(
    fish_data, fish_target, random_state=42)
#%%
print(train_input.shape, train_target.shape)
print(test_input.shape, test_target.shape)

#%%
print(test_target)
#%%
train_input, test_input, train_target, test_target = train_test_split(
    fish_data, fish_target, stratify=fish_target, random_state=42)
#%%
print(test_target)
#%%
from sklearn.neighbors import KNeighborsClassifier
kn = KNeighborsClassifier()
kn.fit(train_input, train_target)
kn.score(test_input, test_target)
#%%
print(kn.predict([[25, 150]]))
#%%
import matplotlib.pyplot as plt
plt.scatter(train_input[:,0], train_input[:,1])
plt.scatter(25, 150, marker='^')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()
#%%
distances, indexes = kn.kneighbors([[25,150]])
#%%
plt.scatter(train_input[:,0], train_input[:,1])
plt.scatter(25, 150, marker='^')
plt.scatter(train_input[indexes,0], train_input[indexes,1], marker='D')
plt.xlabel('length')
plt.ylabel('weight')
#%%
plt.scatter(train_input[:,0], train_input[:,1])
plt.scatter(25, 150, marker='^')
plt.scatter(train_input[indexes,0], train_input[indexes,1], marker='D')
plt.ylim((0, 1000))
plt.xlabel('length')
plt.ylabel('weight')
#%%
train_input
#%%
test_input
test_target
#%%
train_input
train_input[:,0]
train_input[:,1]
#%%
mean = np.mean(train_input, axis=0)
std = np.std(train_input, axis=0)

#%%
print(mean, std)
#train_input 의 평균과 표준편차

train_input

#%%
# 표준 점수 (z score)
train_scaled = (train_input - mean) / std
train_scaled
#%%
# 표준점수로 다시 산점도로 그려보기
plt.scatter(train_scaled[:,0], train_scaled[:,1])
plt.scatter(25, 150, marker='^')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()
# 수상한 도미도 표준점수로 ㄱㄱ
#%%
# 수상한 도미 다시 표시하기
# 수상한 도미의 표준점수(z score)
new = ([25, 150] - mean) / std

plt.scatter(train_scaled[:,0], train_scaled[:,1])
plt.scatter(new[0], new[1], marker='^')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

# 스케일빼고 데이터 모양은 같음.
#%%
# 다시 모델 훈련.
kn.fit(train_scaled, train_target)

# 테스트 스케일 변환

test_scaled = (test_input - mean) / std

# 다시 평가 score
 
kn.score(test_scaled, test_target)

# 1.0

#%%
# 다시 수상한 도미 예측. predict

print(kn.predict([new]))

# [1.]

#%%
# distance indexes 로 가까운 데이터 사용 확인

distances, indexes = kn.kneighbors([new])
plt.scatter(train_scaled[:,0], train_scaled[:,1])
plt.scatter(new[0], new[1], marker='^')
plt.scatter(train_scaled[indexes,0], train_scaled[indexes,1], marker='D')
plt.show()

# 가장 가까운 샘플은 모두 도미인 것을 확인 할 수 있음.

#%%
# stratify = True -- 같은 비율로 data를 train, test 로 나눔

# random_state = -- usually 42. seed 가 42인 shuffle. 