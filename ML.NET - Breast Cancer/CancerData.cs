using Microsoft.ML.Runtime.Api;
using System;
using System.Collections.Generic;
using System.Text;

namespace ML.NET___Breast_Cancer
{
    class CancerData
    {

        [Column(ordinal: "0")]
        public string Diagnosis;

        [Column(ordinal: "1")]
        public float RadiusMean;

        [Column(ordinal: "2")]
        public float TextureMean;

        [Column(ordinal: "3")]
        public float PerimeterMean;

        [Column(ordinal: "4")]
        public float AreaMean;

        [Column(ordinal: "5")]
        public float SmoothnessMean;

        [Column(ordinal: "6")]
        public float CompactnessMean;

        [Column(ordinal: "7")]
        public float ConcavityMean;

        [Column(ordinal: "8")]
        public float ConcavePointsMean;

        [Column(ordinal: "9")]
        public float SymmetryMean;

        [Column(ordinal: "10")]
        public float FractialDimensionMean;

        [Column(ordinal: "11")]
        public float RadiusSe;

        [Column(ordinal: "12")]
        public float TextureSe;

        [Column(ordinal: "13")]
        public float PerimeterSe;

        [Column(ordinal: "14")]
        public float AreaSe;

        [Column(ordinal: "15")]
        public float SmoothnessSe;

        [Column(ordinal: "16")]
        public float CompactnessSe;

        [Column(ordinal: "17")]
        public float ConcavitySe;

        [Column(ordinal: "18")]
        public float ConcavePointsSe;

        [Column(ordinal: "19")]
        public float SymmetrySe;

        [Column(ordinal: "20")]
        public float FractalDimensionSe;

        [Column(ordinal: "21")]
        public float RadiusWorst;

        [Column(ordinal: "22")]
        public float TextureWorst;

        [Column(ordinal: "23")]
        public float PerimeterWorst;

        [Column(ordinal: "24")]
        public float AreaWorst;

        [Column(ordinal: "25")]
        public float SmoothnessWorst;

        [Column(ordinal: "26")]
        public float CompactnessWorst;

        [Column(ordinal: "27")]
        public float ConcavityWorst;

        [Column(ordinal: "28")] 
        public float ConcavPointsWorst;

        [Column(ordinal: "29")]
        public float SymmetryWorst;

        [Column(ordinal: "30")]
        public float FractalDimensionWorst;

        [Column(ordinal: "31", name: "Label")]
        public string Label;


    }
}
