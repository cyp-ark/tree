# Decision Tree and its Ensemble Model, Random Forests

## 1. Introduction
시간이 지남에 따라 다양한 머신러닝 알고리즘이 제안되어 왔지만, 절대적으로 성능이 우수한 모델이 무엇이냐 라고 물어본다면 제대로 대답할 수 있는 사람은 아마 없을 것이다. 물론 보편적으로 어떤 모델이 다른 모델보다 대체적으로 성능이 좋게 나온다 라고 경험적으로 어느정도는 말할 수 있겠지만, 어떤 데이터셋에 적용하느냐에 따라 절대적으로 어떠한 모델이 항상 우수하지는 않을 것이다. 그러나 이러한 성능의 편차를 줄여주게 되는 방법이 제안되었는데 바로 여러 모델을 결합하는 앙상블 학습(Ensemble Learning)이다. 

앙상블 학습은 bagging과 boosting이라는 두가지 방법이 있다. Bagging 계열의 대표적인 모델에는 Random Forests가 있고, boosting 계열에는 XGBoost 등이 있다. 이번 튜토리얼에는 bagging 계열의 Random Forests에 대해 알아보고, 해당 모델의 base learner인 Decision Tree를 bagging해서 직접 구현해보는 시간을 가지려고 한다.

## 2. Method

### 1) Bagging

Bagging은 앙상블 학습 방법 중 하나로 어떠한 데이터 셋을 샘플링 해 여러 모델을 학습한 후 해당 모델들의 결과를 결합하는 방법입니다. Bagging은 크게 두가지 단계로 나누어져 있는데 bootstrap을 이용한 샘플링 단계, 그리고 여러 모델들의 결과를 결합하는 aggregating 단계 입니다.

<p align="center"> <img src="https://github.com/cyp-ark/tree/blob/main/figure/figure2.png?raw=true" width="30%" height="30%">

<p align="center"> Bagging 방법 도식도


**Bootstrap**은 전체 N개의 데이터셋에서 샘플을 그 크기가 N개가 될 때 까지 복원추출하는 방식입니다. 이렇게 샘플링을 할 경우 샘플링이 되지 않은 데이터가 존재하게 되는데 이들의 집합을 Out of Bag(OOB) 데이터라고 합니다. 이론적으로 어떠한 샘플이 bootstrap을 통해 샘플링 되지 않을 확률 $p$는 다음과 같습니다.
   
   
$$
p=(1-\frac{1}{N})^N \rightarrow \lim_{N\rightarrow \infty}(1-\frac{1}{N})^N = e^{-1} \approx 0.368
$$
   
   
**Aggregating**은 bootstrap을 통해 생성된 데이터들을 개별의 여러 모델에 학습 시킨 후 그 결과들을 합쳐주는 과정입니다. Aggregating 방법에는 대표적으로 3가지가 있는데, 첫번째로는 어떠한 값을 예측할 때 가장 많은 모델이 예측한 값을 사용하는 majority voting, 두번째로는 majority voting에서 해당 값의 예측확률이나 해당 모델의 accuracy 등을 가중치로 곱하여 보정하는 weighted voting, 마지막으로는 base learner들이 예측한 값들을 새로운 모델의 input로 넣어 최종 값을 출력하는 stacking이 있습니다.

<p align="center"> <img src="https://github.com/cyp-ark/tree/blob/main/figure/figure1.png?raw=true" width="60%" height="60%">

### 2) Decision Tree

Decision Tree는 데이터를 2개 혹은 그 이상으로 분할해 비슷한 범주끼리 최대한 많이 모이게 나누는 모델입니다. 그 생김새가 마치 나무를 뒤집어 놓은듯 하다 하여 Decision Tree라 불립니다. 분기를 했을 때 최대한 같은 범주끼리 모여있어야 하기 때문에 모든 변수를 대상으로 gini index, cross entrophy, information gain 등을 기준으로 분기할 변수를 정해 분할하는 과정을 거칩니다.


<p align="center"> <img src="https://github.com/cyp-ark/tree/blob/main/figure/figure3.png?raw=true" width="40%" height="40%" >

<p align="center"> Decision Tree 예시



### 3) Random Forests

Random Forests는 base learner로 Decision Tree를 사용해 bagging을 통해 만든 앙상블 모델입니다. Bootstrap을 통해 다양한 샘플을 만들고 개별 모델을 학습 시킨 후 각 모델의 결과를 aggregate해 해당 모델의 최종 output을 산출하게 됩니다.

<p align="center"> <img src="https://github.com/cyp-ark/tree/blob/main/figure/figure4.png?raw=true" width="60%" height="60%">

Random Forests가 모델의 다양성을 확보하기 위해 사용한 방법은 첫번째로 bootstrap을 통한 데이터의 다양성, 두번째로는 분기에 대한 변수 제한입니다. 분기에 대한 변수 제한의 경우 기존의 Decision Tree의 경우 모든 변수에 대해 Information gain을 계산하고 그 중 분기할 변수를 정하게 되는데, 분기에 대한 후보 변수의 수를 제한하는 것 입니다. 이를 통해 같은 tree가 아닌 다양한 tree들을 앙상블 학습에 이용할 수 있습니다.

## 3. Tutorial

### 1) Data description

간단한 데이터 셋을 Decision Tree를 통해 분류해보도록 하겠습니다. 데이터 셋은 다음과 같습니다.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

X, y = datasets.make_gaussian_quantiles(n_samples=500,n_classes=2,random_state=4)

df = pd.DataFrame(X,columns=['x1','x2'])
df['class'] = y

plt.scatter('x1','x2',c='class',data=df)
```
<p align="center"> <img src="https://github.com/cyp-ark/tree/blob/main/figure/plot1.png?raw=true" width="40%" height="40%" >

### 2) Decision Tree

```python
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#Decision Tree 생성
clf = tree.DecisionTreeClassifier()

#train, test split
X_train, X_test = train_test_split(df)

y_train = X_train.pop(X_train.columns[-1])
y_test = X_test.pop(X_test.columns[-1])

clf.fit(X_train,y_train)
pred = clf.predict(X_test)

accuracy_score(y_test,pred)
```

<p align="center"> <img src="https://github.com/cyp-ark/tree/blob/main/figure/plot2.png?raw=true" width="40%" height="40%" >


### 3) Random Forests using Decision tree

```python
from sklearn.utils import resample

#Bootstraping
def bootstrap(X):
    Y = resample(X,replace=True,n_samples=len(X))
    return Y
```

```python
def rf(df,T):
    #train, test split
    df_train,df_test = train_test_split(df)

    X_test = df_test
    y_test = X_test.pop(X_test.columns[-1])
    
    
    vote = np.zeros(shape=(T,len(y_test)))


    for i in range(T):
        #bootstrap
        X_train = bootstrap(df_train)
        y_train = X_train.pop(X_train.columns[-1])

        #Decision tree 생성 (최대 sqrt(N)개의 feature 사용)
        globals()["tree{}".format(i)] = tree.DecisionTreeClassifier(max_features='sqrt')
        globals()["tree{}".format(i)].fit(X_train,y_train)

        pred = globals()["tree{}".format(i)].predict(X_test)
        vote[i,:] = pred

    #Majority voting
    maj = []
    for i in range(len(y_test)):
        if vote[:,i].sum() >= T/2:
           maj.append(1)
        else :
            maj.append(0)

    print(accuracy_score(y_test,maj))

```

| N       | 1      | 2      | 3      | 3      | 4      | 5      | 6      | 7      | 8      | 9      | 10     |
|---------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| RF      | 0.9231 | 0.9510 | 0.9720 | 0.9580 | 0.9510 | 0.9790 | 0.9860 | 0.9790 | 0.9301 | 0.9580 | 0.9650 |
| DT      | 0.9018 | 0.9214 | 0.9301 | 0.9058 | 0.9166 | 0.9304 | 0.9293 | 0.9204 | 0.8965 | 0.9126 | 0.9114 |
| std(DT) | 0.0244 | 0.0208 | 0.0176 | 0.0213 | 0.0221 | 0.0232 | 0.0226 | 0.0215 | 0.0204 | 0.0271 | 0.0222 |


개별 Decision Tree들의 accuracy와 이를 앙상블 한 Random Forests의 accuracy를 비교해보자면, 개별 Decision Tree는 $0.9160\pm0.0116$, Random Forests는 $0.9593\pm0.0200$로 개별 모델의 성능보다 이를 앙상블한 모델이 더 성능이 좋아짐을 확인할 수 있다.

## 4. Application

### 1) Breast cancer Winsonsin diagnostic dataset

```python
#Breast cancer Wisconsin diagnostic dataset
load_df = datasets.load_breast_cancer()

data = pd.DataFrame(load_df.data)
feature = pd.DataFrame(load_df.feature_names)
data.columns = feature[0]
target=pd.DataFrame(load_df.target)
target.columns=['target']
df = pd.concat([data,target],axis=1)
```

## 5. Conclusion



## 6. Reference
1. 데이터분석, 머신러닝 정리 노트 - [Chapter 4. 분류] Decision Tree Classifier [[Link]](https://injo.tistory.com/15)

