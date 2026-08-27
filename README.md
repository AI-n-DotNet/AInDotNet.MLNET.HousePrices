# AInDotNet.MLNET.HousePrices

Learn predictive AI with C# and ML.NET by predicting house sale prices using the Ames Housing dataset.

This project is part of the **AInDotNet Predictive AI Lab Series**. The goal is not simply to train a machine-learning model. The goal is to demonstrate how a developer can work through a predictive AI problem systematically:

**Profile → Baseline → Hypothesis → Experiment → Compare → Improve**

---

## What This Lab Teaches

This exercise demonstrates how to:

- profile a real-world dataset before training a model
- understand numeric and categorical features
- identify missing data
- examine numeric feature correlation with the target
- build a baseline ML.NET regression model
- add features incrementally
- measure whether those features improve the model
- add categorical business context
- use ML.NET AutoML to compare regression approaches
- evaluate R², RMSE, MAE, and individual prediction errors
- analyze model performance across price ranges
- save the trained model
- use the model to predict the price of a hypothetical house

The larger lesson is that predictive AI is not just about choosing an algorithm.

Good results usually come from understanding the data, selecting useful features, adding business context, measuring changes, and improving the model methodically.

---

## Why This Lab Matters

The strongest improvement did not come from immediately choosing a more sophisticated algorithm. It came from progressively adding better information to the model. This lab shows why data understanding and feature engineering are central to predictive AI.

---

## Business Question

> Given what we know about a house, what is it likely to sell for?

This is a **regression** problem because the target is a continuous numeric value:

```text
SalePrice
```

---

## Dataset

This project uses the **Ames Housing dataset**, which contains:

- 2,930 houses
- 82 columns
- historical sale prices
- structural characteristics
- neighborhood information
- zoning
- quality ratings
- garage and basement information
- many other housing attributes

The target column is:

```text
SalePrice
```

Example sale-price distribution:

```text
Minimum:    $12,789
Maximum:    $755,000
Average:    $180,796
Median:     $160,000
```

---

## Step 1: Profile the Data

Before training a model, the application analyzes the dataset.

For numeric candidate features it calculates:

- minimum
- maximum
- average
- median
- missing values

For categorical candidate features it reports:

- number of unique values
- missing values
- most common values

This helps give meaning to the raw columns before they become machine-learning features.

---

## Step 2: Examine Feature Correlation

The application calculates Pearson correlation between numeric candidate features and `SalePrice`.

Example:

```text
Overall Qual           0.799
Gr Liv Area            0.707
Garage Cars            0.648
Garage Area            0.640
Total Bsmt SF          0.632
1st Flr SF             0.622
```

Correlation helps identify numeric features that appear to contain useful predictive signal.

**Correlation is not the same as feature importance.** A weakly correlated feature may still help through nonlinear relationships or interactions with other features. Categorical features such as neighborhood are not evaluated by this numeric correlation calculation.

---

## Step 3: Establish a Baseline

### Run A — Basic Numeric Features

Trainer:

```text
SDCA Regression
```

Results:

```text
R²:    0.825
RMSE:  $34,675
MAE:   $23,463
```

The purpose of the baseline is not to produce the best possible model. It gives us something to compare future experiments against.

---

## Step 4: Add More Numeric Features

### Run B — Expanded Numeric Features

Results:

```text
R²:    0.850
RMSE:  $32,055
MAE:   $21,248
```

Compared with Run A:

```text
R²:    0.825 -> 0.850
RMSE:  $34,675 -> $32,055
MAE:   $23,463 -> $21,248
```

### Lesson

Adding more useful numeric information improved the model, but the improvement was modest.

---

## Step 5: Add Categorical Business Context

### Run C — Numeric + Categorical Features

The model now includes features such as:

- Neighborhood
- MS Zoning
- House Style
- Building Type
- Exterior Quality
- Kitchen Quality
- Basement Quality
- Garage Quality
- Foundation

Results:

```text
R²:    0.892
RMSE:  $27,165
MAE:   $17,852
```

### Lesson

Adding categorical business context produced a larger improvement than simply adding more numeric measurements.

---

## Step 6: Let AutoML Improve the Algorithm

### Run D — Full Features + AutoML

In one representative run, AutoML selected:

```text
LightGbmRegression
```

Results:

```text
R²:    0.917
RMSE:  $23,935
MAE:   $16,275
```

### Lesson

Algorithm selection matters, but much of the improvement had already come from better features and better business context.

---

## Final Experiment Comparison

```text
Run  Description                               R²          RMSE           MAE
-----------------------------------------------------------------------------
A    Basic numeric features                 0.825       $34,675       $23,463
B    Expanded numeric features              0.850       $32,055       $21,248
C    Numeric + categorical features         0.892       $27,165       $17,852
D    Full features + AutoML                 0.917       $23,935       $16,275
```

Baseline to best model:

```text
R² improvement:   +0.092
RMSE reduction:   $10,740
MAE reduction:    $7,188
```

---

## Why Individual Prediction Analysis Matters

Aggregate metrics are useful, but they can hide important failure patterns.

This project also analyzes:

- sample predictions
- best prediction
- worst prediction
- overprediction vs underprediction
- median absolute error
- average percentage error
- error by sale-price range

A model can look good overall while still performing poorly on an important segment of the data.

---

## What We Learned

1. **Understand the data before training a model.**
2. **Correlation can help identify potentially useful numeric features.**
3. **More features do not automatically produce a much better model.**
4. **Categorical and business-context features can be extremely valuable.**
5. **Feature engineering can matter as much as algorithm selection.**
6. **AutoML is useful for comparing trainers and tuning hyperparameters.**
7. **Aggregate metrics can hide important model weaknesses.**
8. **Predictive AI development should be iterative and evidence-driven.**

The workflow demonstrated by this lab is:

```text
Profile
  ↓
Baseline
  ↓
Hypothesis
  ↓
Experiment
  ↓
Compare
  ↓
Improve
```

---

## Hypothetical House Prediction

After training the winning model, the application creates a prediction for a hypothetical house.

Example input:

```text
Living Area:      2,200 sq ft
Bedrooms:         3
Bathrooms:        2
Garage:           2 cars
Year Built:       2005
Overall Quality:  8/10
Neighborhood:     CollgCr
```

Example prediction:

```text
Predicted Price:  approximately $269,000
```

The exact value can vary slightly depending on the AutoML experiment.

---

## Requirements

- Visual Studio 2026 or later
- .NET 10
- ML.NET
- ML.NET AutoML

---

## Running the Project

Clone the repository:

```bash
git clone https://github.com/AI-n-DotNet/AInDotNet.MLNET.HousePrices.git
```

Open the project in Visual Studio.

Make sure the Ames Housing dataset is available at:

```text
Data/AmesHousing.csv
```

Then run the console application.

The program will:

1. profile the dataset
2. analyze candidate features
3. calculate correlations
4. train Run A
5. train Run B
6. train Run C
7. run AutoML
8. compare all experiments
9. save the winning model
10. predict the sale price of a hypothetical house

---

## Try It Yourself

Once you have the project running, try changing one thing at a time.

Ideas:

- remove `OverallQual`
- remove `Neighborhood`
- add another categorical feature
- add another numeric feature
- change the train/test ratio
- increase the AutoML training time
- try another regression trainer manually
- analyze a different price range
- inspect the worst 20 predictions
- build a deliberately weaker model and explain why it performs worse

> Change one major variable at a time so you can understand what caused the result.

---

## Predictive AI Lab Series

This repository is part of a larger series of hands-on C# and ML.NET exercises designed to teach predictive AI fundamentals.

Other labs include:

- Taxi Fare Prediction
- Customer Churn Prediction
- Fraud Detection
- Support Ticket Routing
- Text Classification
- Demand Forecasting
- Sales Anomaly Detection
- Recommendation Systems
- Predictive Maintenance

---

## License

This project is licensed under the MIT License.

---

## About AInDotNet

AInDotNet focuses on practical enterprise AI using C#, .NET, ML.NET, Microsoft Azure, SQL Server, and enterprise application architecture.

The goal is to demonstrate how AI capabilities can be incorporated into real business applications rather than treated as isolated demos.

For more on Predictive AI & Forecasting, visit:
https://aindotnet.com/forecasting/


