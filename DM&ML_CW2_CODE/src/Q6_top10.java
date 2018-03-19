import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Arrays;

//import weka.attributeSelection.AttributeSelection;
import weka.attributeSelection.BestFirst;
import weka.attributeSelection.CfsSubsetEval;
import weka.attributeSelection.CorrelationAttributeEval;
import weka.attributeSelection.InfoGainAttributeEval;
import weka.attributeSelection.Ranker;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.meta.AttributeSelectedClassifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader.ArffReader;
import weka.core.converters.ArffSaver;
import weka.filters.Filter;
import weka.filters.supervised.attribute.*;
import weka.filters.unsupervised.attribute.Remove;

public class Q6_top10 {

	public static void main(String[] args) throws Exception {
		// TODO Auto-generated method stub
		BufferedReader reader =
				   new BufferedReader(new FileReader("fer2017Random.arff"));
				 ArffReader arff = new ArffReader(reader, 1000);
				 Instances data = arff.getStructure();
				 data.setClassIndex(data.numAttributes() - 1);
				 Instance inst;
				 while ((inst = arff.readInstance(data)) != null) {
				   data.add(inst);
				 }
				
				 Remove remove = new Remove();
				 //top 10 attributes from each emotion + class attribute
				 int[] indicesOfColumnsToUse = {1361,1313,1362,1408,1314,1409,1360,1422,1374,1456,23,24,29,25,26,28,30,18,77,78,817,769,865,818,866,910,770,862,913,622,1896,1895,1848,1944,1943,1849,1897,1847,1992,1991,10,11,52,196,88,58,149,244,136,13,550,598,599,647,551,646,600,648,552,549,1407,1455,839,743,744,791,792,1454,2304};
				 
				 remove.setAttributeIndicesArray(indicesOfColumnsToUse);
				 remove.setInvertSelection(true);
				 remove.setInputFormat(data);
				 Instances trainingSubset = Filter.useFilter(data, remove);
				 
				 System.out.println(trainingSubset);
				 
				 
				 
				 Classifier cModel = (Classifier)new NaiveBayes();
				 cModel.buildClassifier(trainingSubset);
				 Evaluation eTest = new Evaluation(trainingSubset);
				 eTest.evaluateModel(cModel, trainingSubset);
				 String strSummary = eTest.toSummaryString();
				 System.out.println(strSummary);
				 System.out.println(eTest.toClassDetailsString());
				 System.out.println(eTest.toMatrixString());
				 
				 ArffSaver saver = new ArffSaver();
			        saver.setInstances(trainingSubset);
			        saver.setFile(new File("fer2017RandomTop10.arff"));
			        saver.writeBatch();
				 
	}

}