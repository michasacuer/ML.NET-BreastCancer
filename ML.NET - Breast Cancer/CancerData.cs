using Microsoft.ML.Data;


namespace ML.NET___Breast_Cancer
{
    class CancerData
    {
        [LoadColumn(1, 30), ColumnName("Features")]
        public float[] FeatureVector { get; set; }

        [LoadColumn(31)]
        public float Target { get; set; }
    }
}
