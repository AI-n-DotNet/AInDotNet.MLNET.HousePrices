using AInDotNet.MLNET.HousePrices.Model;
using System.Globalization;
using System.Text;

namespace AInDotNet.MLNET.HousePrices.Data;

public static class HousingDataLoader
{
    private static readonly string[] RequiredColumns =
    [
        "Overall Qual", "Gr Liv Area", "Garage Cars", "Garage Area", "Total Bsmt SF",
        "1st Flr SF", "Full Bath", "Bedroom AbvGr", "Year Built", "Year Remod/Add",
        "Overall Cond", "Lot Area", "2nd Flr SF", "BsmtFin SF 1", "TotRms AbvGrd",
        "Neighborhood", "MS Zoning", "House Style", "Bldg Type", "Exter Qual",
        "Kitchen Qual", "Bsmt Qual", "Garage Qual", "Foundation", "SalePrice"
    ];

    public static List<HouseData> Load(string filePath)
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
        var columnIndexes = BuildColumnIndex(headers);
        ValidateRequiredColumns(columnIndexes);

        return lines
            .Skip(1)
            .Where(line => !string.IsNullOrWhiteSpace(line))
            .Select(ParseCsvLine)
            .Select(values => CreateHouse(values, columnIndexes))
            .ToList();
    }

    private static Dictionary<string, int> BuildColumnIndex(IReadOnlyList<string> headers)
    {
        var indexes = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);

        for (var i = 0; i < headers.Count; i++)
        {
            indexes[headers[i].Trim()] = i;
        }

        return indexes;
    }

    private static void ValidateRequiredColumns(IReadOnlyDictionary<string, int> columnIndexes)
    {
        foreach (var column in RequiredColumns)
        {
            if (!columnIndexes.ContainsKey(column))
            {
                throw new InvalidOperationException($"Required column '{column}' was not found.");
            }
        }
    }

    private static HouseData CreateHouse(IReadOnlyList<string> values, IReadOnlyDictionary<string, int> columns) => new()
    {
        OverallQual = GetFloat(values, columns["Overall Qual"]),
        GrLivArea = GetFloat(values, columns["Gr Liv Area"]),
        GarageCars = GetFloat(values, columns["Garage Cars"]),
        GarageArea = GetFloat(values, columns["Garage Area"]),
        TotalBsmtSF = GetFloat(values, columns["Total Bsmt SF"]),
        FirstFlrSF = GetFloat(values, columns["1st Flr SF"]),
        FullBath = GetFloat(values, columns["Full Bath"]),
        BedroomAbvGr = GetFloat(values, columns["Bedroom AbvGr"]),
        YearBuilt = GetFloat(values, columns["Year Built"]),
        YearRemodAdd = GetFloat(values, columns["Year Remod/Add"]),
        OverallCond = GetFloat(values, columns["Overall Cond"]),
        LotArea = GetFloat(values, columns["Lot Area"]),
        SecondFlrSF = GetFloat(values, columns["2nd Flr SF"]),
        BsmtFinSF1 = GetFloat(values, columns["BsmtFin SF 1"]),
        TotRmsAbvGrd = GetFloat(values, columns["TotRms AbvGrd"]),
        Neighborhood = GetString(values, columns["Neighborhood"]),
        MSZoning = GetString(values, columns["MS Zoning"]),
        HouseStyle = GetString(values, columns["House Style"]),
        BldgType = GetString(values, columns["Bldg Type"]),
        ExterQual = GetString(values, columns["Exter Qual"]),
        KitchenQual = GetString(values, columns["Kitchen Qual"]),
        BsmtQual = GetString(values, columns["Bsmt Qual"]),
        GarageQual = GetString(values, columns["Garage Qual"]),
        Foundation = GetString(values, columns["Foundation"]),
        SalePrice = GetFloat(values, columns["SalePrice"])
    };

    private static float GetFloat(IReadOnlyList<string> values, int index)
    {
        if (index >= values.Count || IsMissing(values[index]))
        {
            return float.NaN;
        }

        return float.TryParse(values[index].Trim(), NumberStyles.Any, CultureInfo.InvariantCulture, out var result)
            ? result
            : float.NaN;
    }

    private static string GetString(IReadOnlyList<string> values, int index)
    {
        if (index >= values.Count || IsMissing(values[index]))
        {
            return "Missing";
        }

        return values[index].Trim();
    }

    private static bool IsMissing(string? value) =>
        string.IsNullOrWhiteSpace(value) ||
        value.Trim().Equals("NA", StringComparison.OrdinalIgnoreCase) ||
        value.Trim().Equals("N/A", StringComparison.OrdinalIgnoreCase) ||
        value.Trim().Equals("NULL", StringComparison.OrdinalIgnoreCase);

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
