import java.awt.BorderLayout;
import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.BayesNet;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.bayes.net.estimate.SimpleEstimator;
import weka.classifiers.bayes.net.search.local.K2;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader.ArffReader;
import weka.gui.treevisualizer.PlaceNode2;
import weka.gui.treevisualizer.TreeVisualizer;

public class Q8_K2 {

	public static void main(String[] args) throws Exception {
		// TODO Auto-generated method stub
		BufferedReader reader =
				   new BufferedReader(new FileReader("fer2017RandomTop100Att.arff"));
				 ArffReader arff = new ArffReader(reader, 1000);
				 Instances data = arff.getStructure();
				 data.setClassIndex(data.numAttributes() - 1);
				 Instance inst;
				 while ((inst = arff.readInstance(data)) != null) {
				   data.add(inst);
				 }
				 //System.out.println(data.get(0));
				 BayesNet bs = new BayesNet();
				 SimpleEstimator se = new SimpleEstimator();
				 K2 k = new K2();
				 k.setMaxNrOfParents(2);
				 bs.setEstimator(se);
				 bs.setSearchAlgorithm(k);
				 bs.buildClassifier(data);
				 Evaluation eTest = new Evaluation(data);
				 eTest.evaluateModel(bs, data);
				 
				 System.out.println(eTest.toSummaryString());
				 System.out.println(eTest.toClassDetailsString());
				 System.out.println(eTest.toMatrixString());
	}

}