/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package wekamodelimplementation;

import weka.classifiers.trees.J48;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;
import java.util.Random;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;
import java.io.FileInputStream;
import weka.classifiers.Evaluation;
import weka.core.SerializationHelper;
import java.io.File;

/**
 *
 * @author Lenovo
 */
    
public class WekaModelImplementation {

    /**
     * @param args the command line arguments
     */
    public static String[] J48Decisions=new String[34];
    public static String[] NNDecisions=new String[34];
    public static String[] NBDecisions=new String[34];
    public static String[] SVMDecisions=new String[34];
    public static void writeCSV(String container) throws Exception
    {
        BufferedWriter writer = new BufferedWriter(new FileWriter(container));
        for(int i=0;i<34;i++)
        {
            writer.write(J48Decisions[i]+","+NNDecisions[i]+","+SVMDecisions[i]+","+NBDecisions[i]);
            writer.newLine();
        }
        writer.close();
    }
    
    public static void J48Test (Instances test,String model)throws Exception
    {
       System.out.println("Interpreting J48 Model");
       J48 j48;
      j48= (J48)(SerializationHelper.read(new FileInputStream(model)));
      Evaluation eval=new Evaluation(test);
      eval.useNoPriors();
      System.out.println(eval.toMatrixString());
      for(int i=0;i<34;i++)
      {
          eval.evaluateModelOnce(j48, test.get(i));
          if(eval.confusionMatrix()[0][0]==1||eval.confusionMatrix()[1][0]==1)
          {
              J48Decisions[i]="Low";
              System.out.println("Low");
          }
          else
          {
               J48Decisions[i]="High";
              System.out.println("High");
          }
      }
     
    }
    public static void NNTest (Instances test,String model)throws Exception
    {
         System.out.println("Interpreting Neural Network Model");
       MultilayerPerceptron mp;
      mp= (MultilayerPerceptron)(SerializationHelper.read(new FileInputStream(model)));
     
      Evaluation eval=new Evaluation(test);
      eval.useNoPriors();
      System.out.println(eval.toMatrixString());
      for(int i=0;i<34;i++)
      {
          eval.evaluateModelOnce(mp, test.get(i));
          if(eval.confusionMatrix()[0][0]==1||eval.confusionMatrix()[1][0]==1)
          {
              NNDecisions[i]="Low";
              System.out.println("Low");
          }
          else
          {
              NNDecisions[i]="High";
              System.out.println("High");
          }
      }
     
    }
    public static void SVMTest (Instances test,String model)throws Exception
    {
         System.out.println("Interpreting Support Vector Machine Model");
       SMO smo;
      smo= (SMO)(SerializationHelper.read(new FileInputStream(model)));
      Evaluation eval=new Evaluation(test);
      eval.useNoPriors();
      System.out.println(eval.toMatrixString());
      for(int i=0;i<34;i++)
      {
          eval.evaluateModelOnce(smo, test.get(i));
          if(eval.confusionMatrix()[0][0]==1||eval.confusionMatrix()[1][0]==1)
          {
              SVMDecisions[i]="Low";
              System.out.println("Low");
          }
          else
          {
              SVMDecisions[i]="High";
              System.out.println("High");
          }
      }
     
    }
     public static void NBTest (Instances test,String model)throws Exception
    {
         System.out.println("Interpreting Naive Bayes Model");
       NaiveBayes nb;
      nb= (NaiveBayes)(SerializationHelper.read(new FileInputStream(model)));
      Evaluation eval=new Evaluation(test);
      eval.useNoPriors();
      System.out.println(eval.toMatrixString());
      for(int i=0;i<34;i++)
      {
          eval.evaluateModelOnce(nb, test.get(i));
          if(eval.confusionMatrix()[0][0]==1||eval.confusionMatrix()[1][0]==1)
          {
              NBDecisions[i]="Low";
              System.out.println("Low");
          }
          else
          {
              NBDecisions[i]="High";
              System.out.println("High");
          }
      }
     
    }
    public static void main(String[] args) throws Exception{
        System.out.println("Interpreting Models");
        CSVLoader loader = new CSVLoader();
      loader.setSource(new File(args[0]));
      Instances data = loader.getDataSet();
      data.setClassIndex(11);
       ArffSaver saver = new ArffSaver();
    saver.setInstances(data);
    saver.setFile(new File(args[1]));
    saver.writeBatch();
    J48Test(data,args[2]);
    NNTest(data,args[3]);
    SVMTest(data,args[4]);
    NBTest(data,args[5]);
    writeCSV(args[6]);
    }
    
}
