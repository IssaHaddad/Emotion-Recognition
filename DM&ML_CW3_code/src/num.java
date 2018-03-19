import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Random;

import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.ArffLoader.ArffReader;

public class num {

	public static void main(String[] args) throws IOException {
		// TODO Auto-generated method stub
		BufferedReader reader =
				   new BufferedReader(new FileReader("fer2017-testingTop10EachRedInst.arff"));
				 ArffReader arff = new ArffReader(reader, 1000);
				 Instances data = arff.getStructure();
				 data.setClassIndex(data.numAttributes() - 1);
				 Instance inst;
				 int num = 0 ;
				 while ((inst = arff.readInstance(data)) != null) {
				   data.add(inst);
				   num++;
				 }
				 Random rand = new Random();
			     data.randomize(rand);
				 System.out.print(num);
				 
				 ArffSaver saver = new ArffSaver();
			        saver.setInstances(data);
			        saver.setFile(new File("fer2017Random-testingTop10EachRedInst.arff"));
			        saver.writeBatch();
	}

}
