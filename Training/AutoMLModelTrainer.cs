using AInDotNet.MLNET.HousePrices.Evaluation;
using AInDotNet.MLNET.HousePrices.Model;
using Microsoft.ML;
using Microsoft.ML.AutoML;

namespace AInDotNet.MLNET.HousePrices.Training;

public static class AutoMLModelTrainer
{
    public static async Task<ModelTrainingResult> TrainAsync(
        MLContext mlContext,
        IDataView trainData,
        IDataView testData,
        string runId,
        string description)
    {
        Console.WriteLine();
        Console.WriteLine($"Run {runId} - {description}");
        Console.WriteLine(new string('=', 6 + runId.Length + description.Length));
        Console.WriteLine();

        var preprocessing = HousePricePipelineBuilder.CreatePreprocessingPipeline(mlContext, FeatureSet.Full);

        var regressionPipeline = mlContext.Auto().Regression(
            labelColumnName: nameof(HouseData.SalePrice),
            featureColumnName: "Features");

        var pipeline = preprocessing.Append(regressionPipeline);

        mlContext.Log += (_, e) =>
        {
            if (e.Source == nameof(AutoMLExperiment))
            {
                Console.WriteLine(e.RawMessage);
            }
        };

        var experiment = mlContext.Auto().CreateExperiment();
        experiment
            .SetPipeline(pipeline)
            .SetRegressionMetric(
                RegressionMetric.RSquared,
                labelColumn: nameof(HouseData.SalePrice),
                scoreColumn: "Score")
            .SetDataset(trainData, testData)
            .SetTrainingTimeInSeconds(60);

        Console.WriteLine("Running AutoML experiment...");
        Console.WriteLine();

        var result = await experiment.RunAsync();
        var winningPipeline = pipeline.ToString(result.TrialSettings.Parameter);
        var bestModel = result.Model;

        Console.WriteLine();
        Console.WriteLine("Winning AutoML Trial");
        Console.WriteLine("--------------------");
        Console.WriteLine($"Trial ID:   {result.TrialSettings.TrialId}");
        Console.WriteLine($"Metric:     {result.Metric:F4}");
        Console.WriteLine($"Duration:   {result.DurationInMilliseconds:N0} ms");
        Console.WriteLine($"Pipeline:   {winningPipeline}");

        var modelDirectory = Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "Models");
        Directory.CreateDirectory(modelDirectory);

        var modelPath = Path.Combine(modelDirectory, "HousePriceModel.zip");
        mlContext.Model.Save(bestModel, trainData.Schema, modelPath);

        Console.WriteLine();
        Console.WriteLine("Model Saved");
        Console.WriteLine("-----------");
        Console.WriteLine(modelPath);

        var predictions = bestModel.Transform(testData);
        var metrics = mlContext.Regression.Evaluate(
            predictions,
            labelColumnName: nameof(HouseData.SalePrice),
            scoreColumnName: "Score");

        Console.WriteLine();
        Console.WriteLine("AutoML Model Evaluation");
        Console.WriteLine("-----------------------");
        Console.WriteLine($"R-Squared: {metrics.RSquared:F3}");
        Console.WriteLine($"RMSE:      {metrics.RootMeanSquaredError:C0}");
        Console.WriteLine($"MAE:       {metrics.MeanAbsoluteError:C0}");
        Console.WriteLine($"MSE:       {metrics.MeanSquaredError:N0}");

        RegressionPredictionAnalyzer.Analyze(mlContext, predictions, $"Run {runId} Individual Prediction Analysis");

        return new ModelTrainingResult
        {
            RunId = runId,
            Description = description,
            Trainer = winningPipeline,
            FeatureSet = FeatureSet.Full.ToString(),
            Model = bestModel,
            ModelPath = modelPath,
            RSquared = metrics.RSquared,
            RMSE = metrics.RootMeanSquaredError,
            MAE = metrics.MeanAbsoluteError
        };
    }
}
