import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader.ArffReader;

public class Q4 {

	public static void main(String[] args) throws Exception {
		// TODO Auto-generated method stub
		BufferedReader reader =
				   new BufferedReader(new FileReader("fer2017.arff"));
				 ArffReader arff = new ArffReader(reader, 1000);
				 Instances data = arff.getStructure();
				 data.setClassIndex(data.numAttributes() - 1);
				 Instance inst;
				 while ((inst = arff.readInstance(data)) != null) {
				   data.add(inst);
				 }
				 //System.out.println(data.get(0));
				 Classifier cModel = (Classifier)new NaiveBayes();
				 cModel.buildClassifier(data);
				 Evaluation eTest = new Evaluation(data);
				 eTest.evaluateModel(cModel, data);
				 String strSummary = eTest.toSummaryString();
				 System.out.println(strSummary);
				 System.out.println(eTest.toClassDetailsString());
				 System.out.println(eTest.toMatrixString());
	}

}
