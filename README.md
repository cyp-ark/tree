# tree

## 1. Introduction


## 2. Method


### 1) Decision tree


### 2) Bagging


### 3) Random Forests


### 4) Boosting


### 5) Tree-based Gradient Boosting Machine



## 3. Tutorial

### 1) Random Forests using Decision tree

```python
X, y = datasets.make_gaussian_quantiles(n_samples=500,n_classes=2,random_state=4)

df = pd.DataFrame(X,columns=['x1','x2'])
df['class'] = y

plt.scatter('x1','x2',c='class',data=df)
```
<p align="center"> <img src="https://github.com/cyp-ark/tree/blob/main/figure/plot1.png?raw=true" width="40%" height="40%" >

```python
#Bootstraping
def bootstraping(X):
    Y = resample(X,replace=True,n_samples=len(X))
    return Y
```

## 4. Application



## 5. Conclusion



## 6. Reference
1. 데이터분석, 머신러닝 정리 노트 - [Chapter 4. 분류] Decision Tree Classifier [[Link]](https://injo.tistory.com/15)

