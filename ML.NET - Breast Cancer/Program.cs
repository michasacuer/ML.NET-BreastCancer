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
            ;
            //PredictionModel<CancerData, CancerPrediction> model = Train();
            //Evaluate(model);
        }

        //public static PredictionModel<CancerData, CancerPrediction> Train()
        //{
        //    var pipeline = new LearningPipeline();
        //    pipeline.Add(new TextLoader("Cancer-Train.csv").CreateFrom<CancerData>(useHeader: true, separator: ';'));
        //    pipeline.Add(new Dictionarizer(("Diagnosis", "Label")));
        //    pipeline.Add(new ColumnConcatenator(outputColumn: "Features",
        //        "RadiusMean",
        //        "TextureMean",
        //        "PerimeterMean",
        //        "AreaMean",
        //        "SmoothnessMean",
        //        "CompactnessMean",
        //        "ConcavityMean",
        //        "ConcavePointsMean",
        //        "SymmetryMean",
        //        "FractialDimensionMean",
        //        "RadiusSe",
        //        "TextureSe",
        //        "PerimeterSe",
        //        "AreaSe",
        //        "SmoothnessSe",
        //        "CompactnessSe",
        //        "ConcavitySe",
        //        "ConcavePointsSe",
        //        "SymmetrySe",
        //        "FractalDimensionSe",
        //        "RadiusWorst",
        //        "TextureWorst",
        //        "PerimeterWorst",
        //        "AreaWorst",
        //        "SmoothnessWorst",
        //        "CompactnessWorst",
        //        "ConcavityWorst",
        //        "ConcavPointsWorst",
        //        "SymmetryWorst",
        //        "FractalDimensionWorst"));
        //    pipeline.Add(new StochasticDualCoordinateAscentBinaryClassifier());
        //    pipeline.Add(new PredictedLabelColumnOriginalValueConverter() { PredictedLabelColumn = "PredictedLabel" });
        //    PredictionModel<CancerData, CancerPrediction> model = pipeline.Train<CancerData, CancerPrediction>();

        //    model.WriteAsync(modelPath);
        //    return model;

        //}

        //public static void Evaluate(PredictionModel<CancerData, CancerPrediction> model)
        //{
        //    var testData = new TextLoader("Cancer-train.csv").CreateFrom<CancerData>(useHeader: true, separator: ';');
        //    var evaluator = new BinaryClassificationEvaluator();
        //    BinaryClassificationMetrics metrics = evaluator.Evaluate(model, testData);

        //    var accuracy = metrics.Accuracy;
        //    Console.WriteLine("The Accuracy of that model is: " + accuracy);
        //    Console.ReadLine();
        //}
    }
}
