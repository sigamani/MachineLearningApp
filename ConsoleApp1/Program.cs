using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Accord.Controls;
using Accord.IO;
using Accord.Math;
using Accord.Statistics.Distributions.Univariate;
using Accord.MachineLearning.Bayes;
using Accord.MachineLearning.DecisionTrees;
using Accord.MachineLearning.DecisionTrees.Learning;
using System.Data;

namespace ConsoleApp1
{
    class Program
    {
        static void Main(string[] args)
        {
            
            DataTable table = new Accord.IO.CsvReader("C:\\Users\\michael\\Downloads\\JulyToOct2015Test.csv",true).ToTable();

            // Convert the DataTable to input and output vectors
            double[][] inputs = table.ToJagged<double>("BookToPrice", "DividendYield","DebtToEquity", "MarketBeta", "SectorID");
            int[] outputs = table.Columns["MonthlyReturn"].ToArray<int>();
            

            //SecurityID BookToPrice DividendYield EarningsYield   SalesGrowth AssetsToEquity  MarketCap MarketBeta  DebtToEquity    1YrVol  5YrVol  3YrVol ExposureToCurrencyGain  SectorID countryID

            DecisionTree tree = new DecisionTree(
                            inputs: new List<DecisionVariable>
                                {
                        DecisionVariable.Continuous("BookToPrice"),
                        DecisionVariable.Continuous("DividendYield"),
                        DecisionVariable.Continuous("DebtToEquity"),
                        DecisionVariable.Continuous("MarketBeta"),
                        DecisionVariable.Discrete("SectorID", 11)
                        
                                },
                            classes: 2);

            C45Learning teacher = new C45Learning(tree);

            teacher.Learn(inputs, outputs);
            int[] answers = tree.Decide(inputs);


            // Plot the results
           // ScatterplotBox.Show("Expected results", inputs, outputs);
             //ScatterplotBox.Show("Ans", inputs, answers)
            //    .Hold();
        }
    }
}
