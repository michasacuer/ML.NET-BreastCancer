namespace ML.NET___Breast_Cancer
{
    using System;
    using System.Collections.Generic;
    using Microsoft.ML;
    
    public class Program
    {
        public static void Main(string[] args)
        {
            var mlContext = new MLContext();
            var trainData = mlContext.Data.LoadFromTextFile<CancerData>("Cancer-train.csv", hasHeader: true, separatorChar: ';');
            var targetMap = new Dictionary<string, bool> { { "M", true }, { "B", false } };

            var pipeline = mlContext.Transforms.Conversion.MapValue("Target", targetMap)
                .Append(mlContext.Transforms.NormalizeMinMax("Features"))
                .AppendCacheCheckpoint(mlContext)
                .Append(mlContext
                    .BinaryClassification
                    .Trainers
                    .SdcaLogisticRegression(labelColumnName: "Target", featureColumnName: "Features"));

            var model = pipeline.Fit(trainData);

            var testData = mlContext.Data.LoadFromTextFile<CancerData>("Cancer-test.csv", hasHeader: true, separatorChar: ';');
            var metrics = mlContext.BinaryClassification.Evaluate(model.Transform(testData), labelColumnName: "Target");

            Console.WriteLine(metrics.Accuracy);
            Console.ReadLine();
        } 
    }
}
