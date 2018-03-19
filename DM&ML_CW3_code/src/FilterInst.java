import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

import weka.classifiers.rules.ZeroR;
import weka.classifiers.trees.J48;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.ArffLoader.ArffReader;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.RemoveMisclassified;

public class FilterInst {

	public static void main(String[] args) throws Exception {
		// TODO Auto-generated method stub
		BufferedReader reader =
				   new BufferedReader(new FileReader("fer2017-testingTop10Each.arff"));
				 ArffReader arff = new ArffReader(reader, 1000);
				 Instances data = arff.getStructure();
				 data.setClassIndex(data.numAttributes() - 1);
				 Instance inst;
				 while ((inst = arff.readInstance(data)) != null) {
				   data.add(inst);
				 }
				 RemoveMisclassified filter = new RemoveMisclassified();
				 J48 c = new J48();
				 filter.setClassifier(c);
				  filter.setClassIndex(-1);
				  filter.setNumFolds(0);
				  filter.setThreshold(0.1);
				  filter.setMaxIterations(0);
				  filter.setInputFormat(data);
						  for (int i = 0; i < data.numInstances(); i++) {
						    filter.input(data.instance(i));
						  }
						  filter.batchFinished();
						  Instances newData = filter.getOutputFormat();
						  Instance processed;
						  while ((processed = filter.output()) != null) {
						    newData.add(processed);
						  }
						  ArffSaver saver = new ArffSaver();
					        saver.setInstances(newData);
					        saver.setFile(new File("fer2017-testingTop10EachRedInst.arff"));
					        saver.writeBatch();
	}

}
