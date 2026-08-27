using System.Globalization;
using System.Text;

namespace AInDotNet.MLNET.HousePrices.DataAnalysis;

public static class HousingDataProfiler
{
    private static readonly string[] NumericCandidateFeatures =
    [
        "Overall Qual", "Gr Liv Area", "Garage Cars", "Garage Area", "Total Bsmt SF",
        "1st Flr SF", "Full Bath", "Bedroom AbvGr", "Year Built", "Year Remod/Add",
        "Overall Cond", "Lot Area", "2nd Flr SF", "BsmtFin SF 1", "TotRms AbvGrd"
    ];

    private static readonly string[] CategoricalCandidateFeatures =
    [
        "Neighborhood", "MS Zoning", "House Style", "Bldg Type", "Exter Qual",
        "Kitchen Qual", "Bsmt Qual", "Garage Qual", "Foundation"
    ];

    public static void Profile(string filePath)
    {
        if (!File.Exists(filePath))
        {
            throw new FileNotFoundException("Ames Housing CSV file was not found.", filePath);
        }

        var lines = File.ReadAllLines(filePath);

        if (lines.Length < 2)
        {
            throw new InvalidOperationException("The CSV file does not contain any data rows.");
        }

        var headers = ParseCsvLine(lines[0]);
        var rows = lines.Skip(1).Where(line => !string.IsNullOrWhiteSpace(line)).Select(ParseCsvLine).ToList();

        Console.WriteLine("Ames Housing Data Analysis");
        Console.WriteLine("==========================");
        Console.WriteLine();
        Console.WriteLine($"Rows:       {rows.Count:N0}");
        Console.WriteLine($"Columns:    {headers.Count:N0}");
        Console.WriteLine("Target:     SalePrice");
        Console.WriteLine();

        ProfileSalePrice(headers, rows);
        Console.WriteLine();
        ProfileNumericFeatures(headers, rows);
        Console.WriteLine();
        ProfileCategoricalFeatures(headers, rows);
        Console.WriteLine();
        ProfileCorrelations(headers, rows);
        Console.WriteLine();
        ProfileMissingValues(headers, rows);
    }

    private static void ProfileSalePrice(IReadOnlyList<string> headers, IReadOnlyList<List<string>> rows)
    {
        var index = FindColumnIndex(headers, "SalePrice");
        var values = GetNumericValues(rows, index).OrderBy(x => x).ToList();

        Console.WriteLine("Sale Price");
        Console.WriteLine("----------");
        Console.WriteLine($"Minimum:    {values.Min():C0}");
        Console.WriteLine($"Maximum:    {values.Max():C0}");
        Console.WriteLine($"Average:    {values.Average():C0}");
        Console.WriteLine($"Median:     {CalculateMedian(values):C0}");
    }

    private static void ProfileNumericFeatures(IReadOnlyList<string> headers, IReadOnlyList<List<string>> rows)
    {
        Console.WriteLine("Numeric Candidate Features");
        Console.WriteLine("--------------------------");

        foreach (var feature in NumericCandidateFeatures)
        {
            var index = FindColumnIndex(headers, feature);
            var values = GetNumericValues(rows, index).OrderBy(x => x).ToList();
            var missing = rows.Count - values.Count;

            Console.WriteLine();
            Console.WriteLine(feature);
            Console.WriteLine($"  Minimum:  {values.Min():N2}");
            Console.WriteLine($"  Maximum:  {values.Max():N2}");
            Console.WriteLine($"  Average:  {values.Average():N2}");
            Console.WriteLine($"  Median:   {CalculateMedian(values):N2}");
            Console.WriteLine($"  Missing:  {missing:N0}");
        }
    }

    private static void ProfileCategoricalFeatures(IReadOnlyList<string> headers, IReadOnlyList<List<string>> rows)
    {
        Console.WriteLine("Categorical Candidate Features");
        Console.WriteLine("------------------------------");

        foreach (var feature in CategoricalCandidateFeatures)
        {
            var index = FindColumnIndex(headers, feature);
            var nonMissing = rows.Where(row => index < row.Count && !IsMissing(row[index])).Select(row => row[index].Trim()).ToList();
            var missing = rows.Count - nonMissing.Count;
            var topValues = nonMissing.GroupBy(x => x).OrderByDescending(g => g.Count()).Take(5).ToList();

            Console.WriteLine();
            Console.WriteLine(feature);
            Console.WriteLine($"  Unique:   {nonMissing.Distinct(StringComparer.OrdinalIgnoreCase).Count():N0}");
            Console.WriteLine($"  Missing:  {missing:N0}");
            Console.WriteLine("  Most common:");

            foreach (var group in topValues)
            {
                Console.WriteLine($"    {group.Key,-18} {group.Count(),5:N0}");
            }
        }
    }

    private static void ProfileCorrelations(IReadOnlyList<string> headers, IReadOnlyList<List<string>> rows)
    {
        Console.WriteLine("Numeric Feature Correlation with SalePrice");
        Console.WriteLine("------------------------------------------");

        var salePriceIndex = FindColumnIndex(headers, "SalePrice");
        var correlations = new List<(string Feature, double Correlation)>();

        foreach (var feature in NumericCandidateFeatures)
        {
            var featureIndex = FindColumnIndex(headers, feature);
            var paired = new List<(double Feature, double Price)>();

            foreach (var row in rows)
            {
                if (featureIndex >= row.Count || salePriceIndex >= row.Count ||
                    !TryParseDouble(row[featureIndex], out var featureValue) ||
                    !TryParseDouble(row[salePriceIndex], out var price))
                {
                    continue;
                }

                paired.Add((featureValue, price));
            }

            if (paired.Count >= 2)
            {
                correlations.Add((feature, CalculatePearsonCorrelation(paired)));
            }
        }

        foreach (var item in correlations.OrderByDescending(x => Math.Abs(x.Correlation)))
        {
            Console.WriteLine($"{item.Feature,-20} {item.Correlation,7:F3} ({GetCorrelationStrength(item.Correlation)})");
        }

        Console.WriteLine();
        Console.WriteLine("What This Tells Us");
        Console.WriteLine("------------------");
        Console.WriteLine("Correlation measures the strength and direction of a LINEAR relationship");
        Console.WriteLine("between each numeric feature and SalePrice.");
        Console.WriteLine();
        Console.WriteLine("  Near +1 = strong positive linear relationship.");
        Console.WriteLine("  Near  0 = little linear relationship.");
        Console.WriteLine("  Near -1 = strong negative linear relationship.");
        Console.WriteLine();
        Console.WriteLine("Features near the top of this list appear to contain useful predictive signal.");
        Console.WriteLine("For example, Overall Qual and Gr Liv Area move much more closely with SalePrice");
        Console.WriteLine("than Bedroom AbvGr or Overall Cond.");
        Console.WriteLine();
        Console.WriteLine("Important:");
        Console.WriteLine("Correlation is NOT the same as feature importance and does not prove causation.");
        Console.WriteLine("A weakly correlated feature can still help through nonlinear relationships or");
        Console.WriteLine("interactions with other features. Categorical features such as Neighborhood are");
        Console.WriteLine("not evaluated by this numeric Pearson-correlation calculation.");
    }

    private static void ProfileMissingValues(IReadOnlyList<string> headers, IReadOnlyList<List<string>> rows)
    {
        Console.WriteLine("Missing Data");
        Console.WriteLine("------------");

        var missingColumns = new List<(string Column, int Count)>();

        for (var columnIndex = 0; columnIndex < headers.Count; columnIndex++)
        {
            var missing = rows.Count(row => columnIndex >= row.Count || IsMissing(row[columnIndex]));

            if (missing > 0)
            {
                missingColumns.Add((headers[columnIndex], missing));
            }
        }

        foreach (var item in missingColumns.OrderByDescending(x => x.Count).ThenBy(x => x.Column))
        {
            var percentage = rows.Count == 0 ? 0 : (double)item.Count / rows.Count * 100;
            Console.WriteLine($"{item.Column,-22} {item.Count,5:N0} ({percentage,6:F1}%)");
        }

        Console.WriteLine();
        Console.WriteLine("What This Tells Us");
        Console.WriteLine("------------------");
        Console.WriteLine("Missing data deserves investigation, but it does not automatically require deleting");
        Console.WriteLine("rows or filling every value with an average. In housing data, an NA value can sometimes");
        Console.WriteLine("represent the absence of a feature rather than an unknown measurement.");
        Console.WriteLine();
        Console.WriteLine("The modeling features selected for this lab are relatively complete. Several columns");
        Console.WriteLine("with very high missing percentages are not used by the model, so cleaning them would");
        Console.WriteLine("add complexity without necessarily adding predictive value.");
    }

    private static List<double> GetNumericValues(IReadOnlyList<List<string>> rows, int columnIndex) =>
        rows.Where(row => columnIndex >= 0 && columnIndex < row.Count && TryParseDouble(row[columnIndex], out _))
            .Select(row => double.Parse(row[columnIndex], NumberStyles.Any, CultureInfo.InvariantCulture))
            .ToList();

    private static double CalculatePearsonCorrelation(IReadOnlyList<(double Feature, double Price)> values)
    {
        var featureAverage = values.Average(x => x.Feature);
        var priceAverage = values.Average(x => x.Price);
        double covariance = 0, featureVariance = 0, priceVariance = 0;

        foreach (var value in values)
        {
            var featureDifference = value.Feature - featureAverage;
            var priceDifference = value.Price - priceAverage;
            covariance += featureDifference * priceDifference;
            featureVariance += featureDifference * featureDifference;
            priceVariance += priceDifference * priceDifference;
        }

        var denominator = Math.Sqrt(featureVariance * priceVariance);
        return denominator == 0 ? 0 : covariance / denominator;
    }

    private static string GetCorrelationStrength(double correlation)
    {
        var strength = Math.Abs(correlation) switch
        {
            >= 0.80 => "Very Strong",
            >= 0.60 => "Strong",
            >= 0.40 => "Moderate",
            >= 0.20 => "Weak",
            _ => "Very Weak"
        };

        return correlation > 0 ? $"{strength} Positive" : correlation < 0 ? $"{strength} Negative" : "No Linear Correlation";
    }

    private static int FindColumnIndex(IReadOnlyList<string> headers, string columnName)
    {
        for (var i = 0; i < headers.Count; i++)
        {
            if (string.Equals(headers[i].Trim(), columnName, StringComparison.OrdinalIgnoreCase))
            {
                return i;
            }
        }

        throw new InvalidOperationException($"Required column '{columnName}' was not found.");
    }

    private static bool IsMissing(string? value) =>
        string.IsNullOrWhiteSpace(value) ||
        value.Trim().Equals("NA", StringComparison.OrdinalIgnoreCase) ||
        value.Trim().Equals("N/A", StringComparison.OrdinalIgnoreCase) ||
        value.Trim().Equals("NULL", StringComparison.OrdinalIgnoreCase);

    private static bool TryParseDouble(string value, out double result)
    {
        if (IsMissing(value))
        {
            result = 0;
            return false;
        }

        return double.TryParse(value, NumberStyles.Any, CultureInfo.InvariantCulture, out result);
    }

    private static double CalculateMedian(IReadOnlyList<double> sortedValues)
    {
        var middle = sortedValues.Count / 2;
        return sortedValues.Count % 2 == 0 ? (sortedValues[middle - 1] + sortedValues[middle]) / 2.0 : sortedValues[middle];
    }

    private static List<string> ParseCsvLine(string line)
    {
        var values = new List<string>();
        var currentValue = new StringBuilder();
        var insideQuotes = false;

        for (var i = 0; i < line.Length; i++)
        {
            var character = line[i];

            if (character == '"')
            {
                if (insideQuotes && i + 1 < line.Length && line[i + 1] == '"')
                {
                    currentValue.Append('"');
                    i++;
                }
                else
                {
                    insideQuotes = !insideQuotes;
                }
            }
            else if (character == ',' && !insideQuotes)
            {
                values.Add(currentValue.ToString().Trim());
                currentValue.Clear();
            }
            else
            {
                currentValue.Append(character);
            }
        }

        values.Add(currentValue.ToString().Trim());
        return values;
    }
}
