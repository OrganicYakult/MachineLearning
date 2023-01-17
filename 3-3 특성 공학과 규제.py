#%%
# 선형 회귀 = 특성이 한개일 때. target = a(특성) + 절편
# 다중 회귀(여러개의 특성) = 특성이 두개일 때. target = a(특성) + b(특성2) + 절편
# 특성이 세 개일 떄? = 상상 불가(4차원.)
# 농어의 길이, 농어의 높이, 농어의 두께. = 특성 3개
##### 특성 공학 (Feature Engineering)
#
import pandas as pd
# 농어데이터 - length, height, width.
df = pd.read_csv('https://bit.ly/perch_csv_data')
perch_full = df.to_numpy()
print(perch_full)
# pandas로 데이터프레임을 만든 다음 to_numpy()로 넘파이 배열로 바꿈.
#%%
import numpy as np
perch_weight = np.array(
    [5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0, 110.0,
       115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0, 130.0,
       150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0, 197.0,
       218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0, 514.0,
       556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0, 820.0,
       850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0, 1000.0,
       1000.0]
)
#%%
from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(
    perch_full, perch_weight, random_state=42

)
#%%
# 변환기 = transformer
# polynomialFeatures.
# 훈련(fit)을 해야 변환(transform)이 가능하다. 대체로 fit_transform()메서드도 있다.

from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures()
# fit에 들어간 특성 개수만큼 transform에 맞춰야 다항식으로 변함.
poly.fit([[2,3]])
print(poly.transform([[2,3]]))  # [[1. 2. 3. 4. 6. 9.]]
                                # 특성 2개, [2,3]이 6개의 다항식 특성을 가진 샘플.
                                # [1, x, y, xx, xy, yy]
                                # 1은 x*1, y*1 로 항상 들어가있음
                                # 1을 미포함시키려면 include_bias = False
poly = PolynomialFeatures(include_bias = False)
poly.fit([[2,3,4]])
print(poly.transform([[2,3,4]]))  # [[2. 3. 4. 6. 9.]]
#%%
# 이제 데이터 삽입
poly = PolynomialFeatures(include_bias = False)
poly.fit(train_input)
train_poly = poly.transform(train_input)
print(train_input.shape)    # (42, 3)
print(train_poly.shape)     # (42, 9)
print(train_input[0])       # [19.6   5.14  3.04]
print(train_poly[0])        # [ 19.6      5.14     3.04   384.16   100.744   59.584   26.4196  15.6256   9.2416]
                            # [x, y, z, xx, xy, xz, yy, yz, zz]
#%%
poly.get_feature_names_out()

#%%
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(train_poly, train_target)
print(lr.score(train_poly, train_target))
#%%
test_poly = poly.transform(test_input)

print(lr.score(test_poly, test_target))

#%%
poly = PolynomialFeatures(degree=5, include_bias=False)
# degree = int == 고차항. 5라면 5제곱.
poly.fit(train_input)
train_poly = poly.transform(train_input)
test_poly = poly.transform(test_input)
print(train_poly.shape)
poly.get_feature_names_out()
# 특성의 개수가 많을수록 선형 모델은 아주 강력쿠 해진다. == 과대적합.
# 이러면 너무 학습데이터 중심으로 학습하기 때문에 학습데이터는 예측하기 좋지만 실제 데이터는 예측이 안된다.
# 다시 특성을 줄여야하는데 이렇게 규제(regulation)을 쓴다.
#%%
lr.fit(train_poly, train_target)
print(lr.score(train_poly, train_target
))

#%%
# 규제. Regulation
# # 과도한 기울기 조정.
# # 계수값이 너무 크면 규제(Regulation)를 적용하기 전에 정규화를 먼저..(StandardScaler)= 표준 점수
# # ##########################################################이 클래스도 변환기의 하나.
# StandardScaler == 공정하게 데이터값을 사용. = 표준점수

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit(train_poly)
train_scaled = ss.transform(train_poly)
test_scaled = ss.transform(test_poly)

#%%
# 규제 종류 == 
# 릿지 (Ridge)   라쏘 (Lasso)
# Ridge = 계수를 제곱한 값을 기준으로 규제를 적용.
# Lasso = 절댓값을 기준으로 규제를 적용.
#000 릿지를 더 선호.
from sklearn.linear_model import Ridge
ridge = Ridge()
ridge.fit(train_scaled, train_target)
print(ridge.score(train_scaled, train_target))
print(ridge.score(test_scaled, test_target))
#%%
import matplotlib.pyplot as plt
train_score = []
test_score = []
#%%
# alpha == 규제의 강도. 세질수록 과소적합됨.
alpha_list = [0.001, 0.01, 0.1, 1, 10, 100]
for alpha in alpha_list:
    ridge = Ridge(alpha = alpha)
    ridge.fit(train_scaled, train_target)
    train_score.append(ridge.score(train_scaled, train_target))
    test_score.append(ridge.score(test_scaled, test_target))
#%%
print(train_score)
print(test_score)
#%%
plt.plot(np.log10(alpha_list), train_score)
plt.plot(np.log10(alpha_list), test_score)
plt.xlabel('alpha')
plt.ylabel('R^2')
plt.show()
# 두 세트의 점수가 제일 높은 alpha의 값은 -1, 10^-1=0.1 이다.
#%%
# 라쏘 회귀
from sklearn.linear_model import Lasso
lasso = Lasso()
lasso.fit(train_scaled, train_target)
print(lasso.score(train_scaled, train_target))
print(lasso.score(test_scaled, test_target))
#%%
train_score = []
test_score = []
alpha_list
for alpha in alpha_list:
    lasso = Lasso(alpha=alpha, max_iter=10000)
    lasso.fit(train_scaled, train_target)
    train_score.append(lasso.score(train_scaled, train_target))
    test_score.append(lasso.score(test_scaled, test_target))

# ConvergenceWarning: Objective did not converge.....
## max_iter === 반복 횟수.
# max_iter 가 부족할 때 나오는 warning
# 큰 영향을 끼치지 않는다.
#%%
plt.plot(np.log10(alpha_list), train_score)
plt.plot(np.log10(alpha_list), test_score)
plt.xlabel('alpha')
plt.ylabel('R^2')
plt.show()

# 적합한 alpha 값은 1, 즉 10^1 이다.
#%%
lasso = Lasso(alpha=10)
lasso.fit(train_scaled, train_target)
print(lasso.score(train_scaled, train_target))
print(lasso.score(test_scaled, test_target))
#%%
print(np.sum(lasso.coef_ == 0))

#%%
# 라쏘모델은 계수 값을 아예 0으로 만들 수 있음.
# 이걸 lasso.coef_에 저장한다.
print(np.sum(lasso.coef_ == 0))         # 40

# 55개의 특성을 주입했지만 사용한특성은 15개밖에 안됨.
# 결론으로 라쏘는 유용한 특성을 골라내는 용도로도 사용할 수 있다.