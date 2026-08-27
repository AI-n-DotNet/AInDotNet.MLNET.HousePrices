using AInDotNet.MLNET.HousePrices.Evaluation;
using AInDotNet.MLNET.HousePrices.Model;
using Microsoft.ML;

namespace AInDotNet.MLNET.HousePrices.Training;

public static class BaselineModelTrainer
{
    public static ModelTrainingResult Train(MLContext mlContext, IDataView trainData, IDataView testData, FeatureSet featureSet, string runId, string description)
    {
        Console.WriteLine();
        Console.WriteLine($"Run {runId} - {description}");
        Console.WriteLine(new string('=', 6 + runId.Length + description.Length));
        Console.WriteLine();

        var preprocessing = HousePricePipelineBuilder.CreatePreprocessingPipeline(mlContext, featureSet);

        var pipeline = preprocessing.Append(mlContext.Transforms.NormalizeMeanVariance("Features")).Append(mlContext.Regression.Trainers.Sdca(labelColumnName: nameof(HouseData.SalePrice), featureColumnName: "Features"));

        Console.WriteLine("Trainer: SDCA Regression");
        Console.WriteLine($"Features: {GetFeatureSetDescription(featureSet)}");
        Console.WriteLine("Training model...");

        var model = pipeline.Fit(trainData);
        var predictions = model.Transform(testData);
        var metrics = mlContext.Regression.Evaluate(predictions, labelColumnName: nameof(HouseData.SalePrice), scoreColumnName: "Score");

        PrintMetrics(metrics);
        RegressionPredictionAnalyzer.Analyze(mlContext, predictions, $"Run {runId} Individual Prediction Analysis");

        return new ModelTrainingResult
        {
            RunId = runId,
            Description = description,
            Trainer = "SDCA",
            FeatureSet = featureSet.ToString(),
            Model = model,
            RSquared = metrics.RSquared,
            RMSE = metrics.RootMeanSquaredError,
            MAE = metrics.MeanAbsoluteError
        };
    }

    private static void PrintMetrics(Microsoft.ML.Data.RegressionMetrics metrics)
    {
        Console.WriteLine();
        Console.WriteLine("Model Evaluation");
        Console.WriteLine("----------------");
        Console.WriteLine($"R-Squared: {metrics.RSquared:F3}");
        Console.WriteLine($"RMSE:      {metrics.RootMeanSquaredError:C0}");
        Console.WriteLine($"MAE:       {metrics.MeanAbsoluteError:C0}");
        Console.WriteLine($"MSE:       {metrics.MeanSquaredError:N0}");
    }

    private static string GetFeatureSetDescription(FeatureSet featureSet) => featureSet switch
    {
        FeatureSet.BasicNumeric => "10 basic numeric features",
        FeatureSet.EnhancedNumeric => "15 numeric features",
        FeatureSet.Full => "15 numeric + 9 categorical features", _ => featureSet.ToString()
    };
}
