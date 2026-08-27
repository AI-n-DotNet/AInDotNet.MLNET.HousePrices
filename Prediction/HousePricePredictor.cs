using AInDotNet.MLNET.HousePrices.Model;
using Microsoft.ML;

namespace AInDotNet.MLNET.HousePrices.Prediction;

public static class HousePricePredictor
{
    public static float Predict(MLContext mlContext, string modelPath, HouseData house)
    {
        if (!File.Exists(modelPath))
        {
            throw new FileNotFoundException("House price model was not found.", modelPath);
        }

        var model = mlContext.Model.Load(modelPath, out _);
        var predictionEngine = mlContext.Model.CreatePredictionEngine<HouseData, HousePricePrediction>(model);
        return predictionEngine.Predict(house).Score;
    }
}
