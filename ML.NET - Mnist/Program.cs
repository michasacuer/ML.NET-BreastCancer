using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML;
using Accord.DataSets;

namespace ML.NET___Mnist
{
    class Program
    {
        static void Main(string[] args)
        {
            MNIST dataset = new MNIST();
            Console.ReadLine();
        }
    }
}
