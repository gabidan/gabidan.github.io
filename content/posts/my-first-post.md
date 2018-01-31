---
title: "KNN - Python"
date: 2018-01-29T22:46:52Z

---

#### Introduction
	
	
This report tackles a data analysis problem deriving from the Wine Classification data set published by UC Irvine Machine Learning Repository - https://archive.ics.uci.edu/ml/index.php


### Exploratory data analysis I

```
df=pd.read_csv('wineset.csv', header=None)
df.columns=(['Type', 'Alcohol', 'Malic', 'Ash', 
                    'Alcalinity', 'Magnesium', 'Phenols', 
                    'Flavanoids', 'Nonflavanoids',
                    'Proanthocyanins', 'Color', 'Hue', 
                    'Dilution', 'Proline'])
```


```
df.head()
```

<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Type</th>
      <th>Alcohol</th>
      <th>Malic</th>
      <th>Ash</th>
      <th>Alcalinity</th>
      <th>Magnesium</th>
      <th>Phenols</th>
      <th>Flavanoids</th>
      <th>Nonflavanoids</th>
      <th>Proanthocyanins</th>
      <th>Color</th>
      <th>Hue</th>
      <th>Dilution</th>
      <th>Proline</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>14.23</td>
      <td>1.71</td>
      <td>2.43</td>
      <td>15.6</td>
      <td>127</td>
      <td>2.80</td>
      <td>3.06</td>
      <td>0.28</td>
      <td>2.29</td>
      <td>5.64</td>
      <td>1.04</td>
      <td>3.92</td>
      <td>1065</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>13.20</td>
      <td>1.78</td>
      <td>2.14</td>
      <td>11.2</td>
      <td>100</td>
      <td>2.65</td>
      <td>2.76</td>
      <td>0.26</td>
      <td>1.28</td>
      <td>4.38</td>
      <td>1.05</td>
      <td>3.40</td>
      <td>1050</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>13.16</td>
      <td>2.36</td>
      <td>2.67</td>
      <td>18.6</td>
      <td>101</td>
      <td>2.80</td>
      <td>3.24</td>
      <td>0.30</td>
      <td>2.81</td>
      <td>5.68</td>
      <td>1.03</td>
      <td>3.17</td>
      <td>1185</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>14.37</td>
      <td>1.95</td>
      <td>2.50</td>
      <td>16.8</td>
      <td>113</td>
      <td>3.85</td>
      <td>3.49</td>
      <td>0.24</td>
      <td>2.18</td>
      <td>7.80</td>
      <td>0.86</td>
      <td>3.45</td>
      <td>1480</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>13.24</td>
      <td>2.59</td>
      <td>2.87</td>
      <td>21.0</td>
      <td>118</td>
      <td>2.80</td>
      <td>2.69</td>
      <td>0.39</td>
      <td>1.82</td>
      <td>4.32</td>
      <td>1.04</td>
      <td>2.93</td>
      <td>735</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.describe()
```


<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Type</th>
      <th>Alcohol</th>
      <th>Malic</th>
      <th>Ash</th>
      <th>Alcalinity</th>
      <th>Magnesium</th>
      <th>Phenols</th>
      <th>Flavanoids</th>
      <th>Nonflavanoids</th>
      <th>Proanthocyanins</th>
      <th>Color</th>
      <th>Hue</th>
      <th>Dilution</th>
      <th>Proline</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>178.000000</td>
      <td>178.000000</td>
      <td>178.000000</td>
      <td>178.000000</td>
      <td>178.000000</td>
      <td>178.000000</td>
      <td>178.000000</td>
      <td>178.000000</td>
      <td>178.000000</td>
      <td>178.000000</td>
      <td>178.000000</td>
      <td>178.000000</td>
      <td>178.000000</td>
      <td>178.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1.938202</td>
      <td>13.000618</td>
      <td>2.336348</td>
      <td>2.366517</td>
      <td>19.494944</td>
      <td>99.741573</td>
      <td>2.295112</td>
      <td>2.029270</td>
      <td>0.361854</td>
      <td>1.590899</td>
      <td>5.058090</td>
      <td>0.957449</td>
      <td>2.611685</td>
      <td>746.893258</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.775035</td>
      <td>0.811827</td>
      <td>1.117146</td>
      <td>0.274344</td>
      <td>3.339564</td>
      <td>14.282484</td>
      <td>0.625851</td>
      <td>0.998859</td>
      <td>0.124453</td>
      <td>0.572359</td>
      <td>2.318286</td>
      <td>0.228572</td>
      <td>0.709990</td>
      <td>314.907474</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>11.030000</td>
      <td>0.740000</td>
      <td>1.360000</td>
      <td>10.600000</td>
      <td>70.000000</td>
      <td>0.980000</td>
      <td>0.340000</td>
      <td>0.130000</td>
      <td>0.410000</td>
      <td>1.280000</td>
      <td>0.480000</td>
      <td>1.270000</td>
      <td>278.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1.000000</td>
      <td>12.362500</td>
      <td>1.602500</td>
      <td>2.210000</td>
      <td>17.200000</td>
      <td>88.000000</td>
      <td>1.742500</td>
      <td>1.205000</td>
      <td>0.270000</td>
      <td>1.250000</td>
      <td>3.220000</td>
      <td>0.782500</td>
      <td>1.937500</td>
      <td>500.500000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2.000000</td>
      <td>13.050000</td>
      <td>1.865000</td>
      <td>2.360000</td>
      <td>19.500000</td>
      <td>98.000000</td>
      <td>2.355000</td>
      <td>2.135000</td>
      <td>0.340000</td>
      <td>1.555000</td>
      <td>4.690000</td>
      <td>0.965000</td>
      <td>2.780000</td>
      <td>673.500000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>3.000000</td>
      <td>13.677500</td>
      <td>3.082500</td>
      <td>2.557500</td>
      <td>21.500000</td>
      <td>107.000000</td>
      <td>2.800000</td>
      <td>2.875000</td>
      <td>0.437500</td>
      <td>1.950000</td>
      <td>6.200000</td>
      <td>1.120000</td>
      <td>3.170000</td>
      <td>985.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>3.000000</td>
      <td>14.830000</td>
      <td>5.800000</td>
      <td>3.230000</td>
      <td>30.000000</td>
      <td>162.000000</td>
      <td>3.880000</td>
      <td>5.080000</td>
      <td>0.660000</td>
      <td>3.580000</td>
      <td>13.000000</td>
      <td>1.710000</td>
      <td>4.000000</td>
      <td>1680.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 178 entries, 0 to 177
    Data columns (total 14 columns):
    Type               178 non-null int64
    Alcohol            178 non-null float64
    Malic              178 non-null float64
    Ash                178 non-null float64
    Alcalinity         178 non-null float64
    Magnesium          178 non-null int64
    Phenols            178 non-null float64
    Flavanoids         178 non-null float64
    Nonflavanoids      178 non-null float64
    Proanthocyanins    178 non-null float64
    Color              178 non-null float64
    Hue                178 non-null float64
    Dilution           178 non-null float64
    Proline            178 non-null int64
    dtypes: float64(11), int64(3)
    memory usage: 19.5 KB
    


```python
df.isnull().sum()
```




    Type               0
    Alcohol            0
    Malic              0
    Ash                0
    Alcalinity         0
    Magnesium          0
    Phenols            0
    Flavanoids         0
    Nonflavanoids      0
    Proanthocyanins    0
    Color              0
    Hue                0
    Dilution           0
    Proline            0
    dtype: int64




```python
df.hist()
plt.show()
```


![alt text](/images/output_8_0.png)



```python
df.plot(kind='density', subplots=True, layout=(4,4), sharex=False)
plt.show()
```


![alt text](/images/output_9_0.png)



```python
df.plot(kind='box', subplots=True, layout=(3,6), sharex=False, sharey=False)
plt.show()
```


![alt text](/images/output_10_0.png)




* Data frame contains 13 columns and 178 lines
* All data entries are numerical - the category (type) of the cultivar is encoded as: 1, 2 or 3
* Data frame does not have any empty values 
* The first column 'Type' contains the classification of the cultivars
* Min/Max and Mean values observed above suggest that the scales of measure are varied amongst the attributes, therefore, normalizing/standardizing data potentially would be applied in the following steps in order to produce more meaningful results
* 'Type' data is discrete, and the rest of the columns hold continuous data
* It is noticeable that the each of the 3 classes of cultivars ('Type') hold a very similar number of observations



#### Exploratory data analysis II



```python
import seaborn as sns
corr = df[df.columns].corr()
sns.heatmap(corr, cmap="YlGnBu", annot = True)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x2a3309527f0>




![alt text](/images/output_11_1.png)




* It is observed above that cultivar type is least correlated with an attribute 'Ash'
* The highest positive correlation is apparent between cultivar type and: 'Malic', 'Alcalinity', 'Nonflavanoids' and 'Color'
* The highest negative correlation is apparent between cultivar type and: 'Phenols', 'Flavanoids', 'Dilution' 
* It is noticeable that the following attribute pairs are highly positive correlated: 
* Flavanoids and Phenols
* Flavanoids and Dilution
* Color and Alhocol

We have explored the data distribution of the variables in the analysis above where we have identified the attributes which have outliers as well as fairly normal distribution. 




#### The methodology and methods applied

During the exploratory data analysis we have identified that our data set presents a Supervised Learning problem in Machine Learning terms. We are presented with a property (type of the wine cultivar) and our ultimate aim is to be able to predict the type of the cultivar in instances where the cultivar type label is missing. The core idea behind the methodology of our task is to use an algorithm where the set of independent variables (chemical attributes) will be an input into a choice of function which will map the inputs to the desired outputs corresponding to the type of the cultivar. This iterative process will be carried out until the model achieves a desired level of accuracy on the training data which contains information about the true output.  

To narrow it down, our task is to solve a classification problem. A multi-class classification problem, to be precise.


#### KNN


*K- Nearest Neighbours*



Although used for both: classification and regression, KNN is a very popular and simple algorithm mostly adapted in classification problems.

Firstly, the model was implemented using all but one default parameters, such as the following:

* The distance function is defaulted to minkowski, equivalent to the standard Euclidean distance metric (scikit-learn.org) is the most appropriate distance function to use as all our data is numerical.

* Weights are maintained to default allowing all chosen neighbours an equal contribution. Depending on the performance of the model and the investigation into train/test data split, this parameter can be tuned. As our data set is small and generally well distributed (findings from the exploratory data analysis - we have a very similar number of observations in each class) initially weight adjustments are not made.  

* neighbours by default are set to 5. We are working on a small data set, therefore, n is initiated to 3 to avoid overfitting. As this parameter is key in KNN implementation, a cross-validation technique will be used to determine the best number of n. 

Depending on the performance and the nature of this algorithm, the following (in addition to cross-validation) techniques are planned to improve the performance of the model:

* scaling the data 

