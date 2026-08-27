using Microsoft.ML;

namespace AInDotNet.MLNET.HousePrices.Training;

public class ModelTrainingResult
{
    public string RunId { get; set; } = string.Empty;
    public string Description { get; set; } = string.Empty;
    public string Trainer { get; set; } = string.Empty;
    public string FeatureSet { get; set; } = string.Empty;
    public ITransformer? Model { get; set; }
    public string? ModelPath { get; set; }
    public double RSquared { get; set; }
    public double RMSE { get; set; }
    public double MAE { get; set; }
}
