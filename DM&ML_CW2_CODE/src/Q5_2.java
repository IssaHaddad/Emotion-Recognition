import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Arrays;

import weka.attributeSelection.AttributeSelection;
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

public class Q5_2 {

	public static void main(String[] args) throws Exception {
		// TODO Auto-generated method stub
		BufferedReader reader =
				   new BufferedReader(new FileReader("fer2017angry.arff"));
				 ArffReader arff = new ArffReader(reader, 1000);
				 Instances data = arff.getStructure();
				 data.setClassIndex(data.numAttributes() - 1);
				 Instance inst;
				 while ((inst = arff.readInstance(data)) != null) {
				   data.add(inst);
				 }
				 CorrelationAttributeEval eval = new CorrelationAttributeEval();
				 Ranker search = new Ranker();
				 search.setNumToSelect(10);
				 AttributeSelection filter = new AttributeSelection();
				 filter.setEvaluator(eval);
				 filter.setSearch(search);
				 filter.SelectAttributes(data);
				 int[] indices = new int[0];
			     double[][] rankedAttribuesArray = new double[0][0];
			     rankedAttribuesArray = filter.rankedAttributes();
			     indices = filter.selectedAttributes();
			     System.out.println("Top 10 attributes and their corresponding correleation for Angry");
			     System.out.println("note: attributes start from 1 and not from 0");
			     System.out.println("");
			     for(int i =0; i<10; i++)
			     {
			     System.out.println("Attribute: " +((int)rankedAttribuesArray[i][0]+1) +" --> " + rankedAttribuesArray[i][1]);
			     }
			     
	}

}
