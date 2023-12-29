import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVR, SVC
from sklearn.datasets import load_boston
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import scale
import statsmodels.api as sm
import matplotlib.pyplot as plt
import itertools
import time
import seaborn as sns
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense

# 현재 작업 경로 확인.
os.getcwd()
# 데이터 파일 불러오기.
predict=pd.read_csv('C:\\Users\\davye\\num_of_people_add_full_EE.csv')
# 데이터 전처리. 데이터 파일에서 몇가지 독립변수 제외.
predict_=predict.drop(['weather','is_panggah','is_Hoang','time'],axis=1)
# 모델 분석을 위한 데이터 파일에 상수 추가.
predict_data = sm.add_constant(predict_,has_constant='add')
# predict_data의 컬럼들 중에서 종속변수가 아닌가 아닌 컬럼들이 반환. 독립변수들만 저장.
feature_columns = predict_data.columns.difference(['is_Lee'])
# 독립변수들 저장.
X=predict_data[feature_columns]
# 종속 변수 저장.
y=predict_data['is_Lee']
# train data와 test data로 구분. 호출할 때마다 동일한 train data와 test data를 생성하기 위해 난수값 설정.
train_x, test_x, train_y, test_y = train_test_split(X,y, train_size=0.7, test_size=0.3, random_state=1)

# 1. Logistic regression

# 종속 변수("is")를 제외한 독립변수 저장.
feature_columns2 = predict_data.columns.difference(['is_Lee'])
# P-value가 높은 독립변수 제외하고 다시 독립변수 저장.
feature_columns2_=feature_columns2.drop(['Heater2','temp'])
# Logistic regression에 사용할 최종 독립변수 저장.
X=predict_data[feature_columns2_]
# Logistic regression에 사용할 최종 종속변수 저장.
y=predict_data['is_Lee']

# train data와 test data로 구분. 호출할 때마다 동일한 train data와 test data를 생성하기 위해 난수값 설정.
train_x, test_x, train_y, test_y = train_test_split(X,y, train_size=0.7, test_size=0.3, random_state=1)

# 연산 시작 시간.
start=time.time()
# Logistic regression 예측 모델 구축.
logistic_model =sm.Logit(train_y,train_x)
# Logistic regression 예측 모델 적합.
logistic_model=logistic_model.fit(method="newton")
# 적합한 모델에 test data를 넣었을 때의 예측 값 구함.
pred_y = logistic_model.predict(test_x)
# 연산 종료 시간.
end=time.time()
# 연산 시간 출력.
print("[Logistic regression] Time : ",end-start)
# Logistic regression 모델의 결과 요약본을 출력.
#logistic_model.summary()
# 연속된 값을 0과 1로 반환하기 위한 함수.
def cut_off(y,threshold):
    # y값을 복사하는 매서드.
    Y = y.copy()
    # 0과 1을 구분하는 기준점 (threshhold)를 기준으로 1과 0을 구분.
    Y[Y>threshold]=1
    Y[Y<=threshold]=0
    # 정수형으로 Y 값을 반환.
    return(Y.astype(int))
# cut_off 함수를 이용하여 test data로 예측한 연속된 값을 0과 1로 저장.
pred_Y = cut_off(pred_y,0.7)
# 예측 성능을 실제 결과와 비교하여 성능 평가.
cfmat=confusion_matrix(test_y,pred_Y)
# ROC(Receiver operating characteristic) 곡선을 그리기 위한 함수.
fpr1, tpr1, thresholds = metrics.roc_curve(test_y,pred_Y,pos_label=1)
# ROC 곡선 출력.
plt.plot(fpr1,tpr1)
# ROC 곡선 아래 면적 계산.
auc = np.trapz(tpr1,fpr1)
# ROC 곡선 아래 면적 출력.
print('[Logistic regression] AUC:', auc)
# 예측 결과와 실제 결과를 비교하여 Logistic regression의 정확도 계산.
acc = accuracy_score(test_y,pred_Y)
# 정확도 출력.
print('[Logistic regression] Accuracy score : ',acc)