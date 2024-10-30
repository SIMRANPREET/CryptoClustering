# Module 11 Challenge

## Crypto Predictions

This notebook analyses cryptocurrency data.

### Requirements

#### Find the Best Value for k Using the Original Scaled DataFrame

* Code the elbow method algorithm to find the best value for k. Use a range from 1 to 11.

``` python
k = list(range(1, 11))
```

* Visually identify the optimal value for k by plotting a line chart of all the inertia values computed with the different values of k.

``` python
inertia = []
for i in k:
    k_model = KMeans(n_clusters=i, n_init='auto', random_state=1)
    k_model.fit(market_data_scaled_df)
    inertia.append(k_model.inertia_)
elbow_data = {"k": k, "inertia": inertia}
df_elbow = pd.DataFrame(elbow_data)
df_elbow.plot.line(x="k",
                   y="inertia",
                   title="Elbow Curve",
                   xticks=k)
```

* Answer the following question: What’s the best value for k?

``` python
# K = 4 is best since the change in inertia per unit K decreases dramatically after K = 4.
```

#### Cluster Cryptocurrencies with K-Means Using the Original Scaled Data

* Initialize the K-means model with four clusters by using the best value for k.

``` python
k4_model = KMeans(n_clusters=4, n_init='auto', random_state=1)
```

* Fit the K-means model by using the original data

``` python
k4_model.fit(market_data_scaled_df)
```

* Predict the clusters for grouping the cryptocurrencies by using the original data. Review the resulting array of cluster values

``` python
pk4 = k4_model.predict(market_data_scaled_df)
pk4
```

* Create a copy of the original data, and then add a new column of the predicted clusters

``` python
market_data_scaled_predictions_k4_df = market_data_scaled_df.copy()
market_data_scaled_predictions_k4_df["K4_Predictions"] = pk4
```

* Using pandas’ plot, create a scatter plot by setting x="price_change_percentage_24h" and y="price_change_percentage_7d".

``` python
market_data_scaled_predictions_k4_df.plot.scatter(
    x="price_change_percentage_24h", 
    y="price_change_percentage_7d",
    c="K4_Predictions", 
    colormap='rainbow')
```

#### Optimize the Clusters with Principal Component Analysis

* Create a PCA model instance, and set n_components=3 

``` python
pca4_model = PCA(n_components=3)
```

* Use the PCA model to reduce the features to three principal components, then review the first five rows of the DataFrame

``` python
market_data_scaled_pca4 = pca4_model.fit_transform(market_data_scaled_df)
market_data_scaled_pca4[0:5]
```

* Get the explained variance to determine how much information can be attributed to each principal component

``` python
pca4_model.explained_variance_ratio_
```

* Answer the following question: What’s the total explained variance of the three principal components?

``` python
# The total variance that is explained by the three principal components is 89.50%
```

* Create a new DataFrame with the PCA data. Be sure to set the coin_id index from the original DataFrame as the index for the new DataFrame. Review the resulting DataFrame

``` python
market_data_scaled_pca4_df = pd.DataFrame(market_data_scaled_pca4,columns=["PCA1","PCA2","PCA3"])
market_data_scaled_pca4_df.index = market_data_df.index
market_data_scaled_pca4_df.head()
```

#### Find the Best Value for k by Using the PCA Data

* Code the elbow method algorithm, and use the PCA data to find the best value for k. Use a range from 1 to 11.

``` python
k = list(range(1,11))
```

* Visually identify the optimal value for k by plotting a line chart of all the inertia values computed with the different values of k.

``` python
inertia = []
for i in k:
    k_model = KMeans(n_clusters=i, n_init='auto', random_state=1)
    k_model.fit(market_data_scaled_pca4_df)
    inertia.append(k_model.inertia_)
elbow_data = {"k":k,"inertia":inertia}
df_elbow = pd.DataFrame(elbow_data)
df_elbow.plot.line(x="k",
                   y="inertia",
                   title="Elbow Curve",
                   xticks=k)
```

* Answer the following questions: What’s the best value for k when using the PCA data? Does it differ from the best value for k that you found by using the original data?

``` python
# K = 4 is best since the change in inertia per unit K decreases dramatically after K = 4.
# No, there is no difference between the pca data and original data for which K is better. This means that the data can be described by using the pca data thereby reducing the volume of data and increasing efficiency.
```

#### Cluster the Cryptocurrencies with K-Means by Using the PCA Data

* Initialize the K-means model with four clusters by using the best value for k

``` python
k4_model = KMeans(n_clusters=4, n_init='auto', random_state=1)
```

* Fit the K-means model by using the PCA data

``` python
k4_model.fit(market_data_scaled_pca4_df)
```

* Predict the clusters for grouping the cryptocurrencies by using the PCA data. Review the resulting array of cluster values

``` python
pk4 = k4_model.predict(market_data_scaled_pca4_df)
pk4
```

* Create a copy of the DataFrame with the PCA data, and then add a new column to store the predicted clusters

``` python
market_data_scaled_predictions_pca4_df = market_data_scaled_pca4_df.copy()
market_data_scaled_predictions_pca4_df["PCA4_Predictions"] = pk4
```

* Using pandas’ plot, create a scatter plot by setting x="PC1" and y="PC2"

``` python
market_data_scaled_predictions_pca4_df.plot.scatter(
    x="PCA1", 
    y="PCA2",
    c="PCA4_Predictions", 
    colormap='rainbow')
```

#### Determine the Weights of Each Feature on Each Principal Component

* Create a DataFrame that shows the weights of each feature (column) for each principal component by using the columns from the original scaled DataFrame as the index.

``` python
pca_component_weights = pd.DataFrame(pca4_model.components_.T, columns=['PCA1', 'PCA2', 'PCA3'], index=market_data_scaled_df.columns)
pca_component_weights
```

* Answer the following question: Which features have the strongest positive or negative influence on each component?

``` python
"""
For PCA1: the strongest positive influence is by the price_change_percentage_200d and strongest negative influence is price_change_percentage_24h

For PCA2: the strongest positive influence is by the price_change_percentage_30d and strongest negative influence is price_change_percentage_1y

For PCA3: the strongest positive influence is by the price_change_percentage_7d and strongest negative influence is price_change_percentage_60d
"""
```
