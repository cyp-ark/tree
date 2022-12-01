# tree

## 1. Introduction


## 2. Method


### 1) Decision tree


### 2) Bagging


### 3) Random Forests


<!--
### 4) Boosting


### 5) Tree-based Gradient Boosting Machine
-->


## 3. Tutorial

### 1) Data description

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


```python
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



### 2) Random Forests using Decision tree

```python
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

