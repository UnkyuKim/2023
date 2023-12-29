# 필요한 라이브러리 불러오기
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import metrics,svm, datasets
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
import statsmodels.api as sm
from sklearn.svm import SVR
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import r2_score
import time
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.optimizers import SGD

#현재 작업 경로 확인
os.getcwd()
#데이터 불러오기
predict=pd.read_csv('C:\\Users\\kimun\\data\\2023_4\\is_people_add_full_EE .csv')
# 데이터 전처리. 데이터 파일에서 필요없는 독립변수 삭제.
predict_data=predict.drop(['Time','Heater2','Temp','Weather', 'People'],axis=1)


# 1.Multi linear regression


#  P-value가 높은 데이터 독립변수에서 제외.
x_data=predict_data.drop(['People_A','People_B','People_C'],axis=1)
# 종속 변수 지정.
y_data=predict[['Total_consumption']]
# 모델 분석을 위한 상수 추가.
X_data=sm.add_constant(x_data,has_constant='add')
# 독립 변수 지정.
X = X_data
# 종속 변수 지정.
y = y_data

# train data와 test data로 구분. 호출할 때마다 동일한 train data와 test data를 생성하기 위해 난수값 설정.
xtrain, xtest, ytrain, ytest = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=1)
# 시작 시간 측정.
start=time.time()
# train data로 Multi linear regression 예측 모델 구축.
fit_multi_linear=sm.OLS(ytrain, xtrain)
# Multi linear regression 예측 모델 적합.
fit_multi_linear=fit_multi_linear.fit()
# 구축한 예측 모델에 test data를 넣었을 때의 예측값.
y_predict1=fit_multi_linear.predict(xtest)
# 연산 종료 시간.
end=time.time()
# 연산 시간.
print("[Multi linear regression] Time : ",end-start)
# 결정계수로 Multi linear regression 예측 성능 평가.
r2 = r2_score(ytest, y_predict1)
# 예측 성능 출력.
print(f"[Multi linear regression] R-squared : {r2*100:.2f}%")
