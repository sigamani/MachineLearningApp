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
            // Read the Excel worksheet into a DataTable

            DataTable data = new Accord.IO.CsvReader("C:\\Users\\michael\\Downloads\\JulyToOct2015Test.csv",true).ToTable();
            DataTable table = new Accord.IO.ExcelReader("C:\\Users\\michael\\Downloads\\banking.xls").GetWorksheet("banking");

            // Convert the DataTable to input and output vectors
            double[][] inputs = table.ToJagged<double>("X", "Y");
            int[] outputs = table.Columns["G"].ToArray<int>();

            // ScatterplotBox.Show("Yin-Yang", inputs, outputs).Hold();

           // age job marital education   default housing loan contact month day_of_week duration campaign    pdays previous    poutcome emp_var_rate    cons_price_idx cons_conf_idx   euribor3m nr_employed


            DecisionTree tree = new DecisionTree(
                            inputs: new List<DecisionVariable>
                                {
                        DecisionVariable.Discrete("age",),
                        DecisionVariable.Continuous("Y")
                                },
                            classes: 2);

            C45Learning teacher = new C45Learning(tree);

            teacher.Learn(inputs, outputs);
            int[] answers = tree.Decide(inputs);


            // Plot the results
            ScatterplotBox.Show("Expected results", inputs, outputs);
            ScatterplotBox.Show("Naive Bayes results", inputs, answers)
                .Hold();
        }
    }
}
