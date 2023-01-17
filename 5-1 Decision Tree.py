#%%
import pandas as pd
wine = pd.read_csv('https://bit.ly/wine_csv_data')
#%%
wine.head()
#%%
wine.info()
#%%
wine.describe()

# 0 = red wine
# 1 = white wine

#%%
data = wine[['alcohol', 'sugar', 'pH']].to_numpy()
target = wine[['class']].to_numpy()
#%%
from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(
    data, target, test_size=0.2, random_state=42

)
#%%
print(train_input)
#%%
print(train_input.shape, test_input.shape)
#%%
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)
#%%
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(train_scaled, train_target)
print(lr.score(train_scaled, train_target))
print(lr.score(test_scaled, test_target))
#%%
print(lr.coef_, lr.intercept_)
#%%
# 더 쉽게 설명하기.
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(random_state = 42)
dt.fit(train_scaled, train_target)
print(dt.score(train_scaled, train_target))
print(dt.score(test_scaled, test_target))
#%%
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
plt.figure(figsize=(10,7))
plot_tree(dt)
plt.show()
#%%
plt.figure(figsize=(10,7))
plot_tree(dt, max_depth=1, filled=True, feature_names=['alcohol', 'sugar','pH'
])
plt.show()
#%%
plt.figure(figsize=(10,7))
plot_tree(dt, max_depth=1, filled=True, feature_names=['alcohol', 'sugar','pH'
])
plt.show()
# max_depth = *int === *int 행까지 표시.
# filled = *boolean ==== 클래스에 맞게 노드의 색을 칠함.
# feature_names = [''] ===== 특성의 이름을 전달.
# 테스트 조건 (sugar)
# 불순도(gini)
# 총 샘플 수(samples)
# 클래스별 샘플 수(value)
#%%
# 왼쪽이 yes, 오른쪽이 no,  위 테스트 조건에 따라.
# value, 레드와인이 왼쪽값, 화이트화인이 오른쪽 값.
# 오른쪽 비율이 높아질수록 filled 색이 진해짐. (농도)
# 

#%%
# GINI = 불순도
# gini impurity.
# criterion 매개변수의 기본값 == 'gini'
# 지니 불순도 = 1 - (음성비율^2 + 양성비율^2)
# #%%
#%%
dt = DecisionTreeClassifier(max_depth = 3, random_state=42)
dt.fit(train_scaled, train_target)
print(dt.score(train_scaled, train_target))
print(dt.score(test_scaled, test_target))
#%%
plt.figure(figsize=(20,15))
plot_tree(dt, filled=True, feature_names=['alcohol', 'sugar', 'pH'])
plt.show()
#%%
