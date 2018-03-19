import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

import weka.attributeSelection.InfoGainAttributeEval;
import weka.attributeSelection.Ranker;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.meta.AttributeSelectedClassifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader.ArffReader;

public class Q5_1 {

	public static void main(String[] args) throws Exception {
		// TODO Auto-generated method stub
		BufferedReader reader =
				   new BufferedReader(new FileReader("fer2017Random.arff"));
				 ArffReader arff = new ArffReader(reader, 1000);
				 Instances data = arff.getStructure();
				 data.setClassIndex(data.numAttributes() - 1);
				 Instance inst;
				 int counter = 0;
				 while ((inst = arff.readInstance(data)) != null && (counter<(35887/2))) {
				   data.add(inst);
				   counter ++;
				 }
				 Classifier cModel = (Classifier)new NaiveBayes();
				 cModel.buildClassifier(data);
				 Evaluation eTest = new Evaluation(data);
				 eTest.evaluateModel(cModel, data);

				 System.out.println(eTest.toSummaryString());
				 System.out.println(eTest.toClassDetailsString());
				 System.out.println(eTest.toMatrixString());
	}

}
