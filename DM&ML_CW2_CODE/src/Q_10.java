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
import weka.classifiers.bayes.net.search.local.TabuSearch;
import weka.clusterers.ClusterEvaluation;
import weka.clusterers.FarthestFirst;
import weka.clusterers.HierarchicalClusterer;
import weka.clusterers.SimpleKMeans;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader.ArffReader;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
import weka.gui.treevisualizer.PlaceNode2;
import weka.gui.treevisualizer.TreeVisualizer;

public class Q_10 {

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
				
				 // generate data for clusterer (w/o class)
				    Remove filter = new Remove();
				    filter.setAttributeIndices("" + (data.classIndex() + 1));
				    filter.setInputFormat(data);
				    Instances dataClusterer = Filter.useFilter(data, filter);
				 
				 FarthestFirst clusterer = new FarthestFirst();   // new instance of clusterer
				 clusterer.setNumClusters(7);
				 clusterer.buildClusterer(dataClusterer);    // build the clusterer
				 
				// evaluate clusterer
				    ClusterEvaluation eval = new ClusterEvaluation();
				    eval.setClusterer(clusterer);
				    eval.evaluateClusterer(data);

				    // print results
				    System.out.println(eval.clusterResultsToString());
				 
	}

}