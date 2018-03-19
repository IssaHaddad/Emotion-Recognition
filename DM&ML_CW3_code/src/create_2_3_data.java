import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.ArffLoader.ArffReader;

public class create_2_3_data {

	public static void main(String[] args) throws IOException {
		// TODO Auto-generated method stub
		
				 
				 BufferedReader reader =
						   new BufferedReader(new FileReader("fer2017Random-trainingTop10EachRedInst.arff"));
						 ArffReader arff = new ArffReader(reader, 1000);
						 Instances data = arff.getStructure();
						 data.setClassIndex(data.numAttributes() - 1);
						 Instance inst;
						 int num = 0 ;
						 while ((inst = arff.readInstance(data)) != null) {
						   data.add(inst);
						   num++;
						 }
						 System.out.println(num);
						 
						 
						 BufferedReader reader2 =
								   new BufferedReader(new FileReader("fer2017Random-testingTop10EachRedInst.arff"));
								 ArffReader arff2 = new ArffReader(reader2, 1000);
								 Instances data2 = arff2.getStructure();
								 data2.setClassIndex(data2.numAttributes() - 1);
								 Instance inst2;
								 int num2 = 0 ;
								 while ((inst2 = arff2.readInstance(data2)) != null ) {
									 if(num2>=644)
									 {
										 data.add(inst2); 
									 }
									 else
									 {
										 data2.add(inst2);
									 }
								   num2++;
								 }
								 System.out.println(num2);
						 
						 ArffSaver saver = new ArffSaver();
					        saver.setInstances(data);
					        saver.setFile(new File("fer2017BigRandom3-trainingTop10EachRedInst.arff"));
					        saver.writeBatch();
					        
					        ArffSaver saver2 = new ArffSaver();
					        saver2.setInstances(data2);
					        saver2.setFile(new File("fer2017BigRandom3-testingTop10EachRedInst.arff"));
					        saver2.writeBatch();
	}

}
