using Microsoft.ML;
using Microsoft.ML.Data;

namespace AInDotNet.MLNET.HousePrices.Evaluation;

public static class RegressionPredictionAnalyzer
{
    public static void Analyze(MLContext mlContext, IDataView predictions, string title)
    {
        var results = mlContext.Data
            .CreateEnumerable<HousePredictionResult>(predictions, reuseRowObject: false)
            .Select(x => new PredictionAnalysis { ActualPrice = x.SalePrice, PredictedPrice = x.Score })
            .ToList();

        if (results.Count == 0)
        {
            Console.WriteLine("No individual predictions were available.");
            return;
        }

        Console.WriteLine();
        Console.WriteLine(title);
        Console.WriteLine(new string('=', title.Length));
        Console.WriteLine();

        PrintSamplePredictions(results);
        Console.WriteLine();
        PrintBestAndWorstPredictions(results);
        Console.WriteLine();
        PrintPredictionSummary(results);
        PrintPriceBandAnalysis(results);
    }

    private static void PrintSamplePredictions(IReadOnlyList<PredictionAnalysis> results)
    {
        Console.WriteLine("Sample Predictions");
        Console.WriteLine("------------------");
        Console.WriteLine($"{"Actual",12} {"Predicted",12} {"Error",12} {"Error %",10}");
        Console.WriteLine(new string('-', 50));

        foreach (var result in results.Take(10))
        {
            Console.WriteLine($"{result.ActualPrice,12:C0} {result.PredictedPrice,12:C0} {result.SignedError,12:C0} {result.PercentageError,9:F1}%");
        }
    }

    private static void PrintBestAndWorstPredictions(IReadOnlyList<PredictionAnalysis> results)
    {
        Console.WriteLine("Best Prediction");
        Console.WriteLine("---------------");
        PrintPredictionDetails(results.OrderBy(x => x.AbsoluteError).First());

        Console.WriteLine();
        Console.WriteLine("Worst Prediction");
        Console.WriteLine("----------------");
        PrintPredictionDetails(results.OrderByDescending(x => x.AbsoluteError).First());
    }

    private static void PrintPredictionDetails(PredictionAnalysis result)
    {
        Console.WriteLine($"Actual:       {result.ActualPrice:C0}");
        Console.WriteLine($"Predicted:    {result.PredictedPrice:C0}");
        Console.WriteLine($"Error:        {result.SignedError:C0}");
        Console.WriteLine($"Absolute:     {result.AbsoluteError:C0}");
        Console.WriteLine($"Percent:      {result.PercentageError:F1}%");
    }

    private static void PrintPredictionSummary(IReadOnlyList<PredictionAnalysis> results)
    {
        var medianAbsoluteError = CalculateMedian(results.Select(x => x.AbsoluteError).OrderBy(x => x).ToList());

        Console.WriteLine("Prediction Error Summary");
        Console.WriteLine("------------------------");
        Console.WriteLine($"Test Houses:              {results.Count:N0}");
        Console.WriteLine($"Overpredicted:            {results.Count(x => x.SignedError > 0):N0}");
        Console.WriteLine($"Underpredicted:           {results.Count(x => x.SignedError < 0):N0}");
        Console.WriteLine($"Exact Predictions:        {results.Count(x => x.SignedError == 0):N0}");
        Console.WriteLine($"Average Absolute Error:   {results.Average(x => x.AbsoluteError):C0}");
        Console.WriteLine($"Median Absolute Error:    {medianAbsoluteError:C0}");
        Console.WriteLine($"Average Absolute % Error: {results.Average(x => x.AbsolutePercentageError):F1}%");
    }

    private static void PrintPriceBandAnalysis(IReadOnlyList<PredictionAnalysis> results)
    {
        var priceBands = new[]
        {
            new PriceBand("Under $100K", 0, 100_000),
            new PriceBand("$100K - $150K", 100_000, 150_000),
            new PriceBand("$150K - $200K", 150_000, 200_000),
            new PriceBand("$200K - $300K", 200_000, 300_000),
            new PriceBand("$300K - $500K", 300_000, 500_000),
            new PriceBand("$500K and Over", 500_000, double.MaxValue)
        };

        Console.WriteLine();
        Console.WriteLine("Prediction Error by Sale Price Range");
        Console.WriteLine("====================================");
        Console.WriteLine();
        Console.WriteLine($"{"Price Range",-20}{"Houses",8}{"MAE",14}{"Median",14}{"Abs % Err",12}{"Bias",14}");
        Console.WriteLine(new string('-', 82));

        foreach (var band in priceBands)
        {
            var bandResults = results.Where(x => x.ActualPrice >= band.MinimumPrice && x.ActualPrice < band.MaximumPrice).ToList();

            if (bandResults.Count == 0)
            {
                Console.WriteLine($"{band.Name,-20}{0,8}");
                continue;
            }

            var median = CalculateMedian(bandResults.Select(x => x.AbsoluteError).OrderBy(x => x).ToList());

            Console.WriteLine($"{band.Name,-20}{bandResults.Count,8:N0}{bandResults.Average(x => x.AbsoluteError),14:C0}{median,14:C0}{bandResults.Average(x => x.AbsolutePercentageError),11:F1}%{bandResults.Average(x => x.SignedError),14:C0}");
        }

        Console.WriteLine();
        Console.WriteLine("Bias = average predicted price minus actual price.");
        Console.WriteLine("Positive bias means the model tends to overpredict; negative bias means it tends to underpredict.");

        Console.WriteLine();
        Console.WriteLine("Why This Matters");
        Console.WriteLine("----------------");
        Console.WriteLine("Overall model metrics do not tell the entire story.");
        Console.WriteLine("A model can perform well on average while performing poorly for an important");
        Console.WriteLine("segment of the data. Comparing errors by price range helps reveal weaknesses");
        Console.WriteLine("that R², RMSE, and MAE can hide when viewed only as aggregate numbers.");

        var worstBand = priceBands
            .Select(band =>
            {
                var bandResults = results.Where(x => x.ActualPrice >= band.MinimumPrice && x.ActualPrice < band.MaximumPrice).ToList();
                return new { Band = band, Results = bandResults, MAE = bandResults.Count == 0 ? 0 : bandResults.Average(x => x.AbsoluteError) };
            })
            .Where(x => x.Results.Count > 0)
            .OrderByDescending(x => x.MAE)
            .FirstOrDefault();

        if (worstBand is not null)
        {
            Console.WriteLine();
            Console.WriteLine($"Largest MAE in this run: {worstBand.Band.Name} at {worstBand.MAE:C0}.");
            Console.WriteLine($"That segment contains {worstBand.Results.Count:N0} test houses, so sample size should");
            Console.WriteLine("also be considered before drawing broad conclusions.");
        }
    }

    private static double CalculateMedian(IReadOnlyList<double> sortedValues)
    {
        if (sortedValues.Count == 0)
        {
            return 0;
        }

        var middle = sortedValues.Count / 2;
        return sortedValues.Count % 2 == 0 ? (sortedValues[middle - 1] + sortedValues[middle]) / 2.0 : sortedValues[middle];
    }

    private sealed class HousePredictionResult
    {
        public float SalePrice { get; set; }

        [ColumnName("Score")]
        public float Score { get; set; }
    }

    private sealed class PredictionAnalysis
    {
        public double ActualPrice { get; init; }
        public double PredictedPrice { get; init; }
        public double SignedError => PredictedPrice - ActualPrice;
        public double AbsoluteError => Math.Abs(SignedError);
        public double PercentageError => ActualPrice == 0 ? 0 : SignedError / ActualPrice * 100.0;
        public double AbsolutePercentageError => Math.Abs(PercentageError);
    }

    private sealed record PriceBand(string Name, double MinimumPrice, double MaximumPrice);
}
