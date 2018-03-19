import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

import weka.classifiers.Evaluation;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.trees.LMT;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader.ArffReader;

public class Multilayer_perceptron_training {

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
				 
				 
				 MultilayerPerceptron cModel = new MultilayerPerceptron();
				 cModel.setHiddenLayers("10");
					//cModel.setLearningRate(0.1);
					//cModel.setMomentum(0.1);
					//cModel.setValidationThreshold(20);
					//cModel.setTrainingTime(500);
				 cModel.buildClassifier(data);
				 Evaluation eTest = new Evaluation(data);
				 eTest.evaluateModel(cModel, data);

				 System.out.println(eTest.toSummaryString());
				 System.out.println(eTest.toClassDetailsString());
				 System.out.println(eTest.toMatrixString());
				 
				 MultilayerPerceptron cModel2 = new MultilayerPerceptron();
					cModel2.setHiddenLayers("10");
					//cModel2.setLearningRate(0.1);
					//cModel2.setMomentum(0.1);
					//cModel2.setValidationThreshold(20);
					//cModel2.setTrainingTime(500);
					 cModel2.buildClassifier(data);
					 Evaluation evaluation = new Evaluation(data);
					 evaluation.evaluateModel(cModel2, data2);
					 System.out.println(evaluation.toSummaryString());
					 System.out.println(evaluation.toClassDetailsString());
					 System.out.println(evaluation.toMatrixString());
	}

}
