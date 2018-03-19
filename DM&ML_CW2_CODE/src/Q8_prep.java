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

public class Q8_prep {

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
				 CorrelationAttributeEval eval = new CorrelationAttributeEval();
				 Ranker search = new Ranker();
				 search.setNumToSelect(100);
				 AttributeSelection filter = new AttributeSelection();
				 filter.setEvaluator(eval);
				 filter.setSearch(search);
			     
				 filter.setInputFormat(data);
				 Instances newData = Filter.useFilter(data, filter);
				 
				 
				 ArffSaver saver = new ArffSaver();
			        saver.setInstances(newData);
			        saver.setFile(new File("fer2017RandomTop100Att.arff"));
			        saver.writeBatch();
				 
	}

}