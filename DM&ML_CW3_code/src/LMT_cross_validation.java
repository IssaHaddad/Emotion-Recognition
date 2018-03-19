import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Random;

import weka.attributeSelection.InfoGainAttributeEval;
import weka.attributeSelection.Ranker;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.meta.AttributeSelectedClassifier;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.LMT;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader.ArffReader;

public class LMT_cross_validation {

	public static void main(String[] args) throws Exception {
		// TODO Auto-generated method stub
		
		BufferedReader reader2 =
				   new BufferedReader(new FileReader("fer2017Random-testingTop10EachRedInst.arff"));
				 ArffReader arff2 = new ArffReader(reader2, 1000);
				 Instances data2 = arff2.getStructure();
				 data2.setClassIndex(data2.numAttributes() - 1);
				 Instance inst2;
				 while ((inst2 = arff2.readInstance(data2)) != null) {
				   data2.add(inst2);
				 }
		
		BufferedReader reader =
				   new BufferedReader(new FileReader("fer2017Random-trainingTop10EachRedInst.arff"));
				 ArffReader arff = new ArffReader(reader, 1000);
				 Instances data = arff.getStructure();
				 data.setClassIndex(data.numAttributes() - 1);
				 Instance inst;
				 while ((inst = arff.readInstance(data)) != null ) {
				   data.add(inst);
				 }
				 LMT cModel = new LMT();
				 Evaluation eTest = new Evaluation(data);
				 eTest.crossValidateModel(cModel, data, 10, new Random(1));
				 

				 System.out.println(eTest.toSummaryString());
				 System.out.println(eTest.toClassDetailsString());
				 System.out.println(eTest.toMatrixString());
				 
				 LMT cModel2 = new LMT();
				//cModel2.setBinarySplits(true);
				 //cModel2.setConfidenceFactor(0.1);
				// cModel2.setMinNumObj(2);
				 //cModel2.setUnpruned(true);
				 cModel2.buildClassifier(data);
				 Evaluation evaluation = new Evaluation(data);
				 evaluation.evaluateModel(cModel2, data2);
				 System.out.println(evaluation.toSummaryString());
				 System.out.println(evaluation.toClassDetailsString());
				 System.out.println(evaluation.toMatrixString());
	}

}
