using AInDotNet.MLNET.HousePrices.Data;
using AInDotNet.MLNET.HousePrices.DataAnalysis;
using AInDotNet.MLNET.HousePrices.Model;
using AInDotNet.MLNET.HousePrices.Prediction;
using AInDotNet.MLNET.HousePrices.Training;
using Microsoft.ML;

var dataPath = Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "Data", "AmesHousing.csv");

// Step 1: Understand the raw data before training a model.
HousingDataProfiler.Profile(dataPath);

// Step 2: Load only the fields used by this teaching lab.
var houses = HousingDataLoader.Load(dataPath);

Console.WriteLine();
Console.WriteLine($"Loaded {houses.Count:N0} houses for modeling.");

// Step 3: Create one reproducible train/test split and reuse it for every experiment.
var mlContext = new MLContext(seed: 42);
var data = mlContext.Data.LoadFromEnumerable(houses);
var split = mlContext.Data.TrainTestSplit(data, testFraction: 0.20, seed: 42);

var experimentResults = new List<ModelTrainingResult>();

// Step 4: Run controlled experiments. Each run changes one major modeling decision.
PrintExperimentQuestion("Run A", "Establish a simple regression baseline using 10 obvious numeric housing features.");

var runA = BaselineModelTrainer.Train(mlContext, split.TrainSet, split.TestSet, FeatureSet.BasicNumeric, "A", "Basic numeric features");
experimentResults.Add(runA);

PrintExperimentQuestion("Run B", "Do five additional numeric features improve prediction accuracy?");

var runB = BaselineModelTrainer.Train(mlContext, split.TrainSet, split.TestSet, FeatureSet.EnhancedNumeric, "B", "Expanded numeric features");
experimentResults.Add(runB);

ExperimentReporter.PrintChange(runA, runB, "Adding five numeric features improved the model. This tells us that additional useful information can help, but simply adding more numeric columns does not guarantee a dramatic improvement.");

PrintExperimentQuestion("Run C", "Does adding categorical business context such as neighborhood, zoning, house style, and construction quality improve predictions?");

var runC = BaselineModelTrainer.Train(mlContext, split.TrainSet, split.TestSet, FeatureSet.Full, "C", "Numeric + categorical features");
experimentResults.Add(runC);

ExperimentReporter.PrintChange(runB, runC, "Adding categorical business context produced a larger improvement. The model benefited from information that raw numeric measurements alone did not capture.");

PrintExperimentQuestion("Run D", "After improving the features, can AutoML find a better regression algorithm and hyperparameters?");

var runD = await AutoMLModelTrainer.TrainAsync(mlContext, split.TrainSet, split.TestSet, "D", "Full features + AutoML");
experimentResults.Add(runD);

ExperimentReporter.PrintChange(runC, runD, "AutoML improved the model further. Algorithm selection matters, but much of the improvement had already come from giving the model better features and business context.");

ExperimentReporter.PrintComparison(experimentResults);
PrintWhatWeLearned(runA, runB, runC, runD);

// Step 5: Use the winning AutoML model to predict one hypothetical house.
var hypotheticalHouse = new HouseData
{
    OverallQual = 8,
    GrLivArea = 2_200,
    GarageCars = 2,
    GarageArea = 550,
    TotalBsmtSF = 1_100,
    FirstFlrSF = 1_200,
    FullBath = 2,
    BedroomAbvGr = 3,
    YearBuilt = 2005,
    YearRemodAdd = 2010,
    OverallCond = 7,
    LotArea = 10_000,
    SecondFlrSF = 1_000,
    BsmtFinSF1 = 700,
    TotRmsAbvGrd = 8,
    Neighborhood = "CollgCr",
    MSZoning = "RL",
    HouseStyle = "2Story",
    BldgType = "1Fam",
    ExterQual = "Gd",
    KitchenQual = "Gd",
    BsmtQual = "Gd",
    GarageQual = "TA",
    Foundation = "PConc"
};

if (string.IsNullOrWhiteSpace(runD.ModelPath))
{
    throw new InvalidOperationException("The AutoML model was not saved, so a sample prediction cannot be created.");
}

var predictedPrice = HousePricePredictor.Predict(mlContext, runD.ModelPath, hypotheticalHouse);

Console.WriteLine();
Console.WriteLine("Hypothetical House Prediction");
Console.WriteLine("=============================");
Console.WriteLine();
Console.WriteLine($"Living Area:      {hypotheticalHouse.GrLivArea:N0} sq ft");
Console.WriteLine($"Bedrooms:         {hypotheticalHouse.BedroomAbvGr:N0}");
Console.WriteLine($"Bathrooms:        {hypotheticalHouse.FullBath:N0}");
Console.WriteLine($"Garage:           {hypotheticalHouse.GarageCars:N0} cars");
Console.WriteLine($"Year Built:       {hypotheticalHouse.YearBuilt:N0}");
Console.WriteLine($"Overall Quality:  {hypotheticalHouse.OverallQual:N0}/10");
Console.WriteLine($"Neighborhood:     {hypotheticalHouse.Neighborhood}");
Console.WriteLine();
Console.WriteLine($"Predicted Price:  {predictedPrice:C0}");
Console.WriteLine();
Console.WriteLine("Done.");

static void PrintExperimentQuestion(string runName, string question)
{
    Console.WriteLine();
    Console.WriteLine($"{runName} Experiment Question");
    Console.WriteLine(new string('-', $"{runName} Experiment Question".Length));
    Console.WriteLine(question);
    Console.WriteLine();
}

static void PrintWhatWeLearned(ModelTrainingResult runA, ModelTrainingResult runB, ModelTrainingResult runC, ModelTrainingResult runD)
{
    Console.WriteLine();
    Console.WriteLine("WHAT WE LEARNED");
    Console.WriteLine("===============");
    Console.WriteLine();

    Console.WriteLine("1. Understand the data before training a model.");
    Console.WriteLine("   Profiling gives meaning to the raw columns: ranges, typical values, missing data,");
    Console.WriteLine("   categorical values, and relationships with the value we want to predict.");
    Console.WriteLine();

    Console.WriteLine("2. Correlation helps identify potentially useful numeric features.");
    Console.WriteLine("   Strong linear relationships are useful clues, but correlation is not the same");
    Console.WriteLine("   as feature importance and does not tell the entire modeling story.");
    Console.WriteLine();

    Console.WriteLine("3. More numeric features helped, but the improvement was modest.");
    Console.WriteLine($"   Run A -> Run B: R² {runA.RSquared:F3} -> {runB.RSquared:F3}; MAE {runA.MAE:C0} -> {runB.MAE:C0}.");
    Console.WriteLine();

    Console.WriteLine("4. Business context produced a larger improvement.");
    Console.WriteLine($"   Run B -> Run C: R² {runB.RSquared:F3} -> {runC.RSquared:F3}; MAE {runB.MAE:C0} -> {runC.MAE:C0}.");
    Console.WriteLine("   Neighborhood, zoning, house style, and quality categories gave the model useful");
    Console.WriteLine("   information that the numeric measurements did not fully capture.");
    Console.WriteLine();

    Console.WriteLine("5. Algorithm selection mattered, but it was not the whole story.");
    Console.WriteLine($"   Run C -> Run D: R² {runC.RSquared:F3} -> {runD.RSquared:F3}; MAE {runC.MAE:C0} -> {runD.MAE:C0}.");
    Console.WriteLine("   AutoML improved an already much stronger feature set.");
    Console.WriteLine();

    Console.WriteLine("6. Aggregate metrics can hide important failures.");
    Console.WriteLine("   Looking at individual predictions and price bands shows where the model performs");
    Console.WriteLine("   well and where it still struggles.");
    Console.WriteLine();

    Console.WriteLine("7. Predictive AI development is an iterative engineering process.");
    Console.WriteLine("   Profile -> Baseline -> Hypothesis -> Experiment -> Compare -> Improve");
}
