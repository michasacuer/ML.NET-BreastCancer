using System;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;
using Microsoft.ML.Trainers;
using Microsoft.ML;

namespace ML.NET___Breast_Cancer
{
    class Program
    {

        const string modelPath = "@Model.zip";

        static void Main(string[] args)
        {
            var mlContext = new MLContext();
            var trainData = mlContext.Data.ReadFromTextFile<CancerData>("Cancer-train.csv", hasHeader: true, separatorChar: ';');
            var pipeline = mlContext.Transforms.Normalize("Features")
                .AppendCacheCheckpoint(mlContext)
                .Append(mlContext.BinaryClassification.Trainers.StochasticDualCoordinateAscent(labelColumn: "Target", featureColumn: "Features"));

            var model = pipeline.Fit(trainData);

            var testData = mlContext.Data.ReadFromTextFile<CancerData>("Cancer-test.csv", hasHeader: true, separatorChar: ';');
            var metrics = mlContext.BinaryClassification.Evaluate(model.Transform(testData), label: "Target");
            Console.WriteLine(metrics.Accuracy);
            Console.ReadLine();
        } 
    }
}
