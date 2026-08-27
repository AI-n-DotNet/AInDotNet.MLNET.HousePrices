namespace AInDotNet.MLNET.HousePrices.Training;

public static class ExperimentReporter
{
    public static void PrintChange(ModelTrainingResult previous, ModelTrainingResult current, string interpretation)
    {
        Console.WriteLine();
        Console.WriteLine($"Change from Run {previous.RunId}");
        Console.WriteLine(new string('-', $"Change from Run {previous.RunId}".Length));

        Console.WriteLine($"R²:    {previous.RSquared:F3} -> {current.RSquared:F3}  ({FormatSigned(current.RSquared - previous.RSquared, "F3")})");
        Console.WriteLine($"RMSE:  {previous.RMSE:C0} -> {current.RMSE:C0}  ({FormatCurrencyChange(current.RMSE - previous.RMSE)})");
        Console.WriteLine($"MAE:   {previous.MAE:C0} -> {current.MAE:C0}  ({FormatCurrencyChange(current.MAE - previous.MAE)})");

        Console.WriteLine();
        Console.WriteLine("Result:");
        Console.WriteLine(interpretation);
    }

    public static void PrintComparison(IEnumerable<ModelTrainingResult> results)
    {
        var resultList = results.ToList();

        Console.WriteLine();
        Console.WriteLine("FINAL EXPERIMENT COMPARISON");
        Console.WriteLine("===========================");
        Console.WriteLine();
        Console.WriteLine($"{"Run",-5}{"Description",-34}{"R²",10}{"RMSE",14}{"MAE",14}");
        Console.WriteLine(new string('-', 77));

        foreach (var result in resultList)
        {
            Console.WriteLine($"{result.RunId,-5}{result.Description,-34}{result.RSquared,10:F3}{result.RMSE,14:C0}{result.MAE,14:C0}");
        }

        var winner = resultList.OrderByDescending(x => x.RSquared).FirstOrDefault();

        if (winner is null)
        {
            return;
        }

        Console.WriteLine();
        Console.WriteLine($"Best R²: Run {winner.RunId} - {winner.Description} ({winner.RSquared:F3})");

        var baseline = resultList.FirstOrDefault();

        if (baseline is null || ReferenceEquals(baseline, winner))
        {
            return;
        }

        var rSquaredImprovement = winner.RSquared - baseline.RSquared;
        var maeReduction = baseline.MAE - winner.MAE;
        var rmseReduction = baseline.RMSE - winner.RMSE;

        Console.WriteLine();
        Console.WriteLine("Baseline to Best Model");
        Console.WriteLine("----------------------");
        Console.WriteLine($"R² improvement:   {rSquaredImprovement:+0.000;-0.000;0.000}");
        Console.WriteLine($"RMSE reduction:   {rmseReduction:C0}");
        Console.WriteLine($"MAE reduction:    {maeReduction:C0}");
    }

    private static string FormatSigned(double value, string format) =>
        value > 0 ? $"+{value.ToString(format)}" : value.ToString(format);

    private static string FormatCurrencyChange(double value)
    {
        if (value == 0)
        {
            return "$0";
        }

        var formatted = Math.Abs(value).ToString("C0");
        return value > 0 ? $"+{formatted}" : $"-{formatted}";
    }
}
