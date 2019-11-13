namespace ML.NET___Breast_Cancer
{
    using Microsoft.ML.Data;
    
    public class CancerData
    {
        [LoadColumn(1, 30), ColumnName("Features")]
        public float[] FeatureVector { get; set; }

        [LoadColumn(0)]
        public string Target { get; set; }
    }
}
