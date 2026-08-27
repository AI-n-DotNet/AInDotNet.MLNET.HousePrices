namespace AInDotNet.MLNET.HousePrices.Model;

public class HouseData
{
    // Basic numeric features
    public float OverallQual { get; set; }
    public float GrLivArea { get; set; }
    public float GarageCars { get; set; }
    public float GarageArea { get; set; }
    public float TotalBsmtSF { get; set; }
    public float FirstFlrSF { get; set; }
    public float FullBath { get; set; }
    public float BedroomAbvGr { get; set; }
    public float YearBuilt { get; set; }
    public float YearRemodAdd { get; set; }

    // Additional numeric features
    public float OverallCond { get; set; }
    public float LotArea { get; set; }
    public float SecondFlrSF { get; set; }
    public float BsmtFinSF1 { get; set; }
    public float TotRmsAbvGrd { get; set; }

    // Categorical features
    public string Neighborhood { get; set; } = string.Empty;
    public string MSZoning { get; set; } = string.Empty;
    public string HouseStyle { get; set; } = string.Empty;
    public string BldgType { get; set; } = string.Empty;
    public string ExterQual { get; set; } = string.Empty;
    public string KitchenQual { get; set; } = string.Empty;
    public string BsmtQual { get; set; } = string.Empty;
    public string GarageQual { get; set; } = string.Empty;
    public string Foundation { get; set; } = string.Empty;

    // Target / label
    public float SalePrice { get; set; }
}
