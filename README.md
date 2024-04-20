# 𝙵𝙾𝚁𝙴𝚂𝚃 𝙲𝙾𝚅𝙴𝚁 𝚃𝚈𝙿𝙴 𝙿𝚁𝙴𝙳𝙸𝙲𝚃𝙸𝙾𝙽
![image](https://github.com/Tanwar-12/FOREST-COVER-TYPE-PREDICTION/assets/110081008/864ed676-e849-4910-a1c8-669a32c35ccb)![image](https://github.com/Tanwar-12/FOREST-COVER-TYPE-PREDICTION/assets/110081008/864ed676-e849-4910-a1c8-669a32c35ccb)![image](https://github.com/Tanwar-12/FOREST-COVER-TYPE-PREDICTION/assets/110081008/864ed676-e849-4910-a1c8-669a32c35ccb)

## **1.𝙸𝙽𝚃𝚁𝙾𝙳𝚄𝙲𝚃𝙸𝙾𝙽**: 
*Forests play a crucial role in maintaining ecological balance and supporting diverse ecosystems. Understanding and accurately classifying different types of forest cover is essential for effective land management, conservation efforts, and ecological research. In this project, our aim is to develop a predictive model capable of classifying forest cover types based on various geographic and environmental features.* 
## 2.𝙱𝚄𝚂𝙸𝙽𝙴𝚂𝚂 𝙲𝙰𝚂𝙴:
*The goal of the Project is to predict seven different Cover Types in four different Wilderness Areas of the Roosevelt National Forest of Northern Colorado with the best accuracy.*
## 3.𝙸𝙼𝙿𝙾𝚁𝚃𝙸𝙽𝙶 𝚃𝙷𝙴 𝙿𝚈𝚃𝙷𝙾𝙽 𝙻𝙸𝙱𝚁𝙰𝚁𝙸𝙴𝚂:

* numpy, pandas, sklearn, matplot,seaborn....
  
  ## 4.𝙻𝙾𝙰𝙳𝙸𝙽𝙶 𝚃𝙷𝙴 𝙳𝙰𝚃𝙰𝚂𝙴𝚃
  **data = pd.read_csv('forest_train.csv')**
 ## 5.𝚃𝙰𝚂𝙺: 𝙼𝚄𝙻𝚃𝙸-𝙲𝙻𝙰𝚂𝚂 𝙲𝙻𝙰𝚂𝚂𝙸𝙵𝙸𝙲𝙰𝚃𝙸𝙾𝙽
 ## 6.𝙳𝙾𝙼𝙰𝙸𝙽 𝙰𝙽𝙰𝙻𝚈𝚂𝙸𝚂:
 - Each observation is a 30m x 30m patch of forest that are classified as one of seven `Cover_Type`.
- The `Soil_Type` (40 columns) & `Wilderness_Area` (4 Columns) are One Hot Encoded.
- The first columns is `Id` which needs to be set as index.

 | Name | Measurement | Description |
| --- | --- | --- |
| Elevation | meters | Elevation in meters |
| Aspect | azimuth | Aspect in degrees azimuth |
| Slope | degrees | Slope in degrees |
| Horizontal Distance To Hydrology | meters | Horz Dist to nearest surface water features |
| Vertical Distance To Hydrology | meters | Vert Dist to nearest surface water features |
| Horizontal Distance To Roadways | meters | Horz Dist to nearest roadway |
| Hillshade 9am | 0 to 255 index | Hillshade index at 9am, summer solstice |
| Hillshade Noon | 0 to 255 index | Hillshade index at noon, summer soltice |
| Hillshade 3pm | 0 to 255 index | Hillshade index at 3pm, summer solstice |
| Horizontal Distance To Fire Points | meters | Horz Dist to nearest wildfire ignition points |
| Wilderness Area (4 binary columns) | 0 (absence) or 1 (presence) | Wilderness area designation |
| Soil Type (40 binary columns) | 0 (absence) or 1 (presence) | Soil Type designation |
| Cover Type | Classes 1 to 7 | Forest Cover Type designation - Response Variable |


1. **Elevation** - Elevation in meters
2. **Aspect** - Aspect in degrees azimuth
3. **Slope** - Slope in degrees
4. **Horizontal_Distance_To_Hydrology** - Horz Dist to nearest surface water features
5. **Slope** - Slope in degrees
6. **Horizontal_Distance_To_Hydrology** - Horz Dist to nearest surface water features
7. **Vertical_Distance_To_Hydrology** - Vert Dist to nearest surface water features
8. **Horizontal_Distance_To_Roadways** - Horz Dist to nearest roadway
9. **Hillshade_9am (0 to 255 index)** - Hillshade index at 9am, summer solstice
10. **Hillshade_Noon (0 to 255 index)** - Hillshade index at noon, summer solstice
11. **Hillshade_3pm (0 to 255 index)** - Hillshade index at 3pm, summer solstice
12. **Horizontal_Distance_To_Fire_Points** - Horz Dist to nearest wildfire ignition points
13. **Wilderness_Area (4 binary columns, 0 = absence or 1 = presence)** - Wilderness area designation
14. **Soil_Type (40 binary columns, 0 = absence or 1 = presence)** - Soil Type designation
15. **Cover_Type (7 types, integers 1 to 7)** - Forest Cover Type designation

The Wilderness Areas are:

1. - Rawah Wilderness Area
2. - Neota Wilderness Area
3. - Comanche Peak Wilderness Area
4. - Cache la Poudre Wilderness Area

The Soil Types are:
1. Cathedral family - Rock outcrop complex, extremely stony.
2. Vanet - Ratake families complex, very stony.
3. Haploborolis - Rock outcrop complex, rubbly.
4. Ratake family - Rock outcrop complex, rubbly.
5. Vanet family - Rock outcrop complex complex, rubbly.
6. Vanet - Wetmore families - Rock outcrop complex, stony.
7. Gothic family.
8. Supervisor - Limber families complex.
9. Troutville family, very stony.
10. Bullwark - Catamount families - Rock outcrop complex, rubbly.
11. Bullwark - Catamount families - Rock land complex, rubbly.
12. Legault family - Rock land complex, stony.
13. Catamount family - Rock land - Bullwark family complex, rubbly.
14. Pachic Argiborolis - Aquolis complex.
15. unspecified in the USFS Soil and ELU Survey.
16. Cryaquolis - Cryoborolis complex.
17. Gateview family - Cryaquolis complex.
18. Rogert family, very stony.
19. Typic Cryaquolis - Borohemists complex.
20. Typic Cryaquepts - Typic Cryaquolls complex.
21. Typic Cryaquolls - Leighcan family, till substratum complex.
22. Leighcan family, till substratum, extremely bouldery.
23. Leighcan family, till substratum - Typic Cryaquolls complex.
24. Leighcan family, extremely stony.
25. Leighcan family, warm, extremely stony.
26. Granile - Catamount families complex, very stony.
27. Leighcan family, warm - Rock outcrop complex, extremely stony.
28. Leighcan family - Rock outcrop complex, extremely stony.
29. Como - Legault families complex, extremely stony.
30. Como family - Rock land - Legault family complex, extremely stony.
31. Leighcan - Catamount families complex, extremely stony.
32. Catamount family - Rock outcrop - Leighcan family complex, extremely stony.
33. Leighcan - Catamount families - Rock outcrop complex, extremely stony.
34. Cryorthents - Rock land complex, extremely stony.
35. Cryumbrepts - Rock outcrop - Cryaquepts complex.
36. Bross family - Rock land - Cryumbrepts complex, extremely stony.
37. Rock outcrop - Cryumbrepts - Cryorthents complex, extremely stony.
38. Leighcan - Moran families - Cryaquolls complex, extremely stony.
39. Moran family - Cryorthents - Leighcan family complex, extremely stony.
40. Moran family - Cryorthents - Rock land complex, extremely stony.

#### Dependent/Target Variable
- Cover_Type (7 categories)

## 7.𝙱𝙰𝚂𝙸𝙲 𝙲𝙷𝙴𝙲𝙺𝚂:
* data.shape,data.head(5),data = data.set_index('Id'),data.sample(10)

  ### 𝙴𝚇𝙰𝙼𝙸𝙽𝙴 𝚃𝙷𝙴 𝙳𝙰𝚃𝙰:
  * data.info()
  ### 𝚂𝚃𝙰𝚂𝚃𝙸𝙲𝙰𝙻 𝙼𝙴𝙰𝚂𝚄𝚁𝙴 𝙾𝙵 𝙳𝙰𝚃𝙰
  data.describe(include='all'),Checking the Skewness of all the features

  ## 8.𝙴𝚇𝙿𝙻𝙾𝚁𝙰𝚃𝙾𝚁𝚈 𝙳𝙰𝚃𝙰 𝙰𝙽𝙰𝙻𝚈𝚂𝙸𝚂
  ###  𝚄𝙽𝙸𝚅𝙰𝚁𝙰𝚃𝙴,𝙱𝙸𝚅𝙰𝚁𝙸𝙰𝚃𝙴 & 𝙼𝚄𝙻𝚃𝙸𝚅𝙰𝚁𝙸𝙰𝚃𝙴 𝙰𝙽𝙰𝙻𝚈𝚂𝙸𝚂
  #### 𝙸𝚖𝚙𝚘𝚛𝚝𝚊𝚗𝚝 𝙿𝚕𝚘𝚝𝚜:
  ![image](https://github.com/Tanwar-12/FOREST-COVER-TYPE-PREDICTION/assets/110081008/d456f7ad-6e44-44cf-87dc-153e5d238636)

  #### 𝚁𝙴𝚂𝚄𝙻𝚃:
  - `Type 7` Cover is densely populated in the Highest Elevation Range. (3100m-3500m approx.)
- `Type 4` Cover is densely populated in the Lowest Elevation Range. (2000m-2500m approx.)
- `Type 4`, `Type 5` & `Type 7` Covers almost never Overlap. They exist in separate Elevation Ranges.
- `Type 3` & `Type 6` are populated at almost same Elevation Range.

![image](https://github.com/Tanwar-12/FOREST-COVER-TYPE-PREDICTION/assets/110081008/447649bb-0637-4131-8197-fd1009af2549)

**From the above Pie Chart, we see that**,

- `Wilderness Area 3` has the highest presence in the Dataset. (42%)
- `Wilderness Area 2` has either very low representation in the Dataset or the Area does not occur frequently. (3.3%)

![image](https://github.com/Tanwar-12/FOREST-COVER-TYPE-PREDICTION/assets/110081008/24d0bc0b-ba12-469b-a026-33877e3c21b1)

**The above plot explains that**,

- `Wilderness Area 2` has only 3 Cover Types.
- `Wilderness Area 3` has 6 Cover Types that are almost balanced.
- Cover Type 4 exists almost entirely in `Wilderness Area 4`.
- Cover Types 3 and Cover Type 6 are only found in `Wilderness Area 3` & `Wilderness Area 4`.

## 𝙲𝙾𝚁𝚁𝙴𝙻𝙰𝚃𝙸𝙾𝙽𝚂: 𝙸𝙽𝙸𝚃𝙸𝙰𝙻 𝙾𝙱𝚂𝙴𝚁𝚅𝙰𝚃𝙸𝙾𝙽:
![image](https://github.com/Tanwar-12/FOREST-COVER-TYPE-PREDICTION/assets/110081008/450b8896-03f8-4436-a88d-3a7dfa25b2f3)

## 𝙿𝚕𝚘𝚝𝚝𝚒𝚗𝚐 𝚊𝚋𝚘𝚟𝚎 𝚌𝚘𝚛𝚛𝚎𝚕𝚊𝚝𝚒𝚘𝚗𝚜:
![image](https://github.com/Tanwar-12/FOREST-COVER-TYPE-PREDICTION/assets/110081008/509e56b5-001d-4cf6-b095-4b49c55fbf88)

### 𝙴𝙳𝙰 𝚂𝚄𝙼𝙼𝙰𝚁𝚈:
- The `Wilderness_Area1`, `Soil_Type38` and `Soil_Type39` are top 3 features that show some correlation.
- There are No Strongly correlated features based on the Correlation Bar Plot.
- We have highly skewed data as well.
- The `Soil_Type7` and `Soil_Type15` only contain 0 value it them.

  ## 9. 𝙳𝙰𝚃𝙰 𝙿𝚁𝙴𝙿𝚁𝙾𝙲𝙴𝚂𝚂𝙸𝙽𝙶:
 #### CHECK MISSING VALUES & UNWANTED COLUMNS

 * data.isna().sum().any()
- There are NO missing values.
- All columns seem to be important as of now. No need to delete any columns.

  #### 𝙲𝙷𝙴𝙲𝙺 𝙳𝚄𝙿𝙻𝙸𝙲𝙰𝚃𝙴𝚂:
  * data.duplicated().any()
    - There are NO duplicate rows in this dataset.
  ### 𝙲𝙷𝙴𝙲𝙺 𝙸𝙼𝙱𝙰𝙻𝙰𝙽𝙲𝙴 𝙳𝙰𝚃𝙰:
  * data['Cover_Type'].value_counts()
    ![image](https://github.com/Tanwar-12/FOREST-COVER-TYPE-PREDICTION/assets/110081008/76b11daf-8a26-41b1-8112-df9c7b406467)

- The Dataset is Balanced. No need for sampling.
- Each Cover Types amounts to 14.29% in the Dataset.

  ###  𝙲𝚁𝙴𝙰𝚃𝙸𝙽𝙶 𝙽𝚄𝙼𝙴𝚁𝙸𝙲𝙰𝙻𝚂 & 𝙲𝙰𝚃𝙴𝙶𝙾𝚁𝙸𝙲𝙰𝙻 𝙻𝙸𝚂𝚃𝚂:
  Creating Subsets of Main DataFrame for exploration:

1. `cont_data` - Data without binary features i.e continuous features
2. `binary_data` - Data having all binary features [Wilderness Areas + Soil Types]
3. `Wilderness_data` - Binary Wilderness Areas
4. `Soil_data` - Binary Soil Types
## 𝙲𝙷𝙴𝙲𝙺 𝙾𝚄𝚃𝙻𝙸𝙴𝚁𝚂:
![image](https://github.com/Tanwar-12/FOREST-COVER-TYPE-PREDICTION/assets/110081008/d3c32571-a257-469f-9e4e-9dc8e533f751)
![image](https://github.com/Tanwar-12/FOREST-COVER-TYPE-PREDICTION/assets/110081008/bb389ea8-f771-437a-906a-605848748fb3)

**From the above plots, we see,**

- `Elevation` and `Aspect` do not have any outliers.
- All other features have significant outliers.
- We need to Treat the outliers in order to understand the correlations better and build better models.
- **Tukey's IQR method**

*Tukey’s (1977) technique is used to detect outliers in skewed or non bell-shaped data since it makes no distributional assumptions. However, Tukey’s method may not be appropriate for a small sample size. The general rule is that anything not in the range of (Q1 - 1.5 IQR) and (Q3 + 1.5 IQR) is an outlier, and can be removed.*

*Inter Quartile Range (IQR) is one of the most extensively used procedure for outlier detection and removal.*

**Procedure:**

1. Find the first quartile, Q1.
2. Find the third quartile, Q3.
3. Calculate the IQR. IQR = Q3-Q1.
4. Define the normal data range with lower limit as Q1–1.5 IQR and upper limit as Q3+1.5 IQR.

**Any data point outside this range is considered as outlier and should be removed for further analysis**.

## 𝙲𝙾𝚁𝚁𝙴𝙻𝙰𝚃𝙸𝙾𝙽𝚂 : 𝙵𝙸𝙽𝙰𝙻 𝙾𝙱𝚂𝙴𝚁𝚅𝙰𝚃𝙸𝙾𝙽𝚂:
![image](https://github.com/Tanwar-12/FOREST-COVER-TYPE-PREDICTION/assets/110081008/5ac6885e-bc76-4e54-a2f1-b32e21b5c6c4)



![image](https://github.com/Tanwar-12/FOREST-COVER-TYPE-PREDICTION/assets/110081008/1713f893-dff8-4395-9325-aa46ad5bb0eb)


## 𝙵𝙴𝙰𝚃𝚄𝚁𝙴 𝙴𝙽𝙶𝙸𝙽𝙴𝙴𝚁𝙸𝙽𝙶: 
 #### 𝙲𝚁𝙴𝙰𝚃𝙸𝙽𝙶 𝙽𝚄𝙼𝙴𝚁𝙸𝙲 & 𝙲𝙰𝚃𝙴𝙶𝙾𝚁𝙸𝙲𝙰𝙻 𝙻𝙸𝚂𝚃𝚂 𝙵𝙾𝚁 𝙵𝙴𝙰𝚃𝚄𝚁𝙴 𝚃𝚁𝙰𝙽𝚂𝙵𝙾𝚁𝙼𝙰𝚃𝙸𝙾𝙽:


**Creating a List of Numeric Columns:**

numeric_columns = ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology',
                'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',
                'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
                'Horizontal_Distance_To_Fire_Points']

**We do not have any Categorical Columns**

## 𝚂𝙿𝙻𝙸𝚃𝙸𝙽𝙶 𝚃𝙷𝙴 𝙳𝙰𝚃𝙰:
## 10.𝙵𝙴𝙰𝚃𝚄𝚁𝙴 𝚂𝙲𝙻𝙰𝙸𝙽𝙶
### FEATURE ENCODING:
- The Features are all  `Numeric`.
- `Soil Types` & `Wilderness Areas` are already One Hot Encoded.
- No need for Futher Feature Encoding as of now.
- As our Target Variable is `Numeric`, we don't need to Label Encode it. (If the Target is categorical, LabelEncoder() should be used before the Train-Test Split.)

### FEATURE SELECTION:
- We have selected the Outlier Removed DataFrame: `data_out`
- We have then applied Stratified Train-Test Split.
- `StandardScaler()` has been used for Feature Scaling.
- Now, we will try Pearson Correlation for Feature Selection.

 ##  **Feature Selection**: Based on Pearson Correlation

- We will use the same `X_train` and `y_train` as they contain scaled data.
- We will create new DataFrame with dropped columns based on Low Correlation.

 ### Feature Engineering Summary:

1. We have Raw Data: `X_train` and `X_test`
2. We have Standard Scaler Tranformed Data: `X_train_ss` and `X_test_ss`
3. We have Data selected based on Pearson Correlation: `X_train_cr` and `X_test_cr`
 
## 11.𝙼𝙾𝙳𝙴𝙻 𝙱𝚄𝙸𝙻𝙳𝙸𝙽𝙶 
### 𝙸𝙽𝙸𝚃𝙸𝙰𝙻 𝚃𝙴𝚂𝚃𝙸𝙽𝙶 𝙾𝙽 𝙲𝙻𝙰𝚂𝚂𝙸𝙵𝙸𝙴𝚁𝚂:

 'Algorithm'	                     'Model Accuracy'
0	LinearSVC             	           0.471848
1	DecisionTreeClassifier	           0.763801
2	LogisticRegression	               0.673167
3	GaussianNB	                       0.585004
4	RandomForestClassifier	           0.854985
5	GradientBoostingClassifier	       0.795111
6	KNNeighborsClassifier   	         0.792365

## 𝚅𝙾𝚃𝙸𝙽𝙶 𝙲𝙻𝙰𝚂𝚂𝙸𝙵𝙴𝚁:
 precision    recall  f1-score   support

           1       0.72      0.78      0.75       521
           2       0.74      0.60      0.66       502
           3       0.79      0.83      0.81       528
           4       0.91      0.98      0.94       532
           5       0.89      0.94      0.91       531
           6       0.85      0.77      0.81       536
           7       0.92      0.95      0.94       491

    accuracy                           0.83      3641
    macro avg       0.83      0.83      0.83      3641
    weighted avg       0.83      0.83      0.83      3641
### HYPERPARAMETER TUNING FOR RANDOM FOREST USING GRID SEARCH
* The Training Accuracy is: 0.9530305804797656
* The Testing Accuracy is: 0.8313650096127437
  
## 𝙲𝙾𝙼𝙿𝙰𝚁𝙸𝚂𝙾𝙽 𝙾𝙵 𝙼𝙾𝙳𝙴𝙻:
- Random Forest achieved the highest accuracy: 85.88% (Although, Overfit!)
- Best Parameters after Hyperparameter Tuning for Random Forest:

    ```
    RandomForestClassifier(n_estimators=30,criterion='gini', n_jobs=-1, 
                            max_depth=25, min_samples_leaf=4, max_features='log2', bootstrap=True,
                            random_state=4)
    ```

- Models used for Voting Classifier: `[KNN, LogisticRegression, DecisionTree, RandomForests]`

**Detailed Model Comparison (Criteria: Accuracy)**

| Model | Accuracy on Raw Data | Accuracy on Scaled Data | Accuracy after Removing 5 Features (Pearson Correlation) |
| --- | --- | --- | --- |
| LinearSVC | 0.519637 | 0.589673 | 0.337270 |
| DecisionTreeClassifier | 0.770393 | 0.727547 | 0.768470 |
| LogisticRegression | 0.663279 | 0.644328 | 0.673167 |
| GaussianNB | 0.581159 | 0.581434 | 0.585004 |
| **`RandomForestClassifier`** | 0.855809 | 0.825597 | **`0.858830`** |
| GradientBoostingClassifier | 0.793189 | 0.750893 | 0.795111 |
| KNNeighborsClassifier | 0.792365 | 0.724801 | 0.792365 |
| Voting Classifier ‘Soft Voting’ | 0.815984 | 0.779181 | 0.817632 |
| Voting Classifier ‘Hard Voting’ | 0.836034 | 0.797033 | 0.834660 |

# 𝙼𝙾𝙳𝙴𝙻 𝚂𝙴𝙻𝙴𝙲𝚃𝙸𝙾𝙽 𝚁𝙴𝙿𝙾𝚁𝚃:
**Train_Data Selected:** Pearson Correlation Selected Data.
* It performed well for most Classifiers.

**Model Selected:** **Random Forests Classifier**

**Model Accuracy:**  **82**

**Reasons for Selecting Random Forests:**
- I selected this model because Random Forests does not always require the Data to be Scaled, since it is a Tree-Based algorithm.
- Random Forests also does not enforce Normal Distribution and since our data was highly skewed and highly un-correlated, it becomes a better choice.
- Random Forests provided the best Accuracy among the tested Algorithms.

  ## 𝚁𝙴𝙿𝙾𝚁𝚃 𝙲𝙷𝙰𝙻𝙻𝙴𝙽𝙶𝙴𝚂 𝙵𝙰𝙲𝙴𝙳:
  - When Random Forest gave the highest Accuracy it came at a cost because the model was clearly overfitted.
- It had a Training Accuracy of 100%.
- In order to make a more Generalised Model, without losing any further training accuracy, I decided to prune the trees and further tune the Hyperparameters.
- The challenge was to get better accuracy while maintaining a Generalised Model. So I settled at **82%** Test Accuracy and 91% Training Accuracy, with 655 misclassifications. **This was the best accuracy where the difference between Train & Test Accuracies was minimum.**
- The model seemed to lose accuracy when trained on less features that are not correlated. Top 49 features out of 54 performed the best.

  # 𝙲𝙾𝙽𝙲𝙻𝚄𝚂𝙸𝙾𝙽:
  - Forest Cover Prediction Dataset was both tricky and easy in many ways because it had All-Numeric data with most of the features One-Hot-Encoded and No-Missing values. But it also had very Skewed features with 4.25% Outliers(IQR) and Least Correlated Data. With these datapoints, Random Forests Classifier achieved 82% Accuracy when trained on most important 49 features based on Pearson-r value based selection.



