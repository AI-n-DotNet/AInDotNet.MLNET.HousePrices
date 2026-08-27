using AInDotNet.MLNET.HousePrices.Model;
using Microsoft.ML;
using Microsoft.ML.Transforms;

namespace AInDotNet.MLNET.HousePrices.Training;

public static class HousePricePipelineBuilder
{
    private static readonly string[] BasicNumericFeatures =
    [
        nameof(HouseData.OverallQual),
        nameof(HouseData.GrLivArea),
        nameof(HouseData.GarageCars),
        nameof(HouseData.GarageArea),
        nameof(HouseData.TotalBsmtSF),
        nameof(HouseData.FirstFlrSF),
        nameof(HouseData.FullBath),
        nameof(HouseData.BedroomAbvGr),
        nameof(HouseData.YearBuilt),
        nameof(HouseData.YearRemodAdd)
    ];

    private static readonly string[] AdditionalNumericFeatures =
    [
        nameof(HouseData.OverallCond),
        nameof(HouseData.LotArea),
        nameof(HouseData.SecondFlrSF),
        nameof(HouseData.BsmtFinSF1),
        nameof(HouseData.TotRmsAbvGrd)
    ];

    private static readonly (string Output, string Input)[] CategoricalFeatures =
    [
        ("NeighborhoodEncoded", nameof(HouseData.Neighborhood)),
        ("MSZoningEncoded", nameof(HouseData.MSZoning)),
        ("HouseStyleEncoded", nameof(HouseData.HouseStyle)),
        ("BldgTypeEncoded", nameof(HouseData.BldgType)),
        ("ExterQualEncoded", nameof(HouseData.ExterQual)),
        ("KitchenQualEncoded", nameof(HouseData.KitchenQual)),
        ("BsmtQualEncoded", nameof(HouseData.BsmtQual)),
        ("GarageQualEncoded", nameof(HouseData.GarageQual)),
        ("FoundationEncoded", nameof(HouseData.Foundation))
    ];

    public static IEstimator<ITransformer> CreatePreprocessingPipeline(MLContext mlContext, FeatureSet featureSet)
    {
        var numericFeatures = featureSet == FeatureSet.BasicNumeric
            ? BasicNumericFeatures
            : BasicNumericFeatures.Concat(AdditionalNumericFeatures).ToArray();

        var pipeline = mlContext.Transforms.ReplaceMissingValues(
            numericFeatures.Select(name => new InputOutputColumnPair(name)).ToArray(),
            replacementMode: MissingValueReplacingEstimator.ReplacementMode.Mean);

        if (featureSet != FeatureSet.Full)
        {
            return pipeline.Append(mlContext.Transforms.Concatenate("Features", numericFeatures));
        }

        var oneHotColumns = CategoricalFeatures
            .Select(x => new InputOutputColumnPair(x.Output, x.Input))
            .ToArray();

        var allFeatureColumns = numericFeatures
            .Concat(CategoricalFeatures.Select(x => x.Output))
            .ToArray();

        return pipeline
            .Append(mlContext.Transforms.Categorical.OneHotEncoding(oneHotColumns))
            .Append(mlContext.Transforms.Concatenate("Features", allFeatureColumns));
    }
}
