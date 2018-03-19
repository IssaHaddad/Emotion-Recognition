import java.awt.List;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;

import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;


public class Q2 {

	public static void main(String[] args) throws IOException{
		// TODO Auto-generated method stub
		String csvFile = "fer2017.csv";
        String line = "";
        String ClassName = "";
        String AttributeName = "";
        int AttSize = 0;
        int[] Class = new int[35887];
        int[][] Attribute = new int[35887][2304];
        int C =0;
        int A =0;
        try (BufferedReader br = new BufferedReader(new FileReader(csvFile))) {

        	String[] first = br.readLine().split(",");
    		ClassName = first[0];
    		AttributeName = first[1];
        	
            while ((line = br.readLine()) != null) {
            		String[] str1 = line.split(",");
            	    String[] att = str1[1].split(" ");
            	    AttSize = att.length;
            	    Class[C] = Integer.parseInt(str1[0]);
            	    for (int i =0 ; i<att.length; i++)
            	    {
            	    	Attribute[C][i] = Integer.parseInt(att[i]);
            	    }
            	    C++;
            }

        } catch (IOException e) {
            e.printStackTrace();
        }
        //System.out.println(Class.length);
        ArrayList<Attribute> atts = new ArrayList<Attribute>();
        ArrayList<String> attVals = new ArrayList<String>();
        Instances data;
        double[] vals;
        
        for(int i=0; i<AttSize; i++)
        {
        	atts.add(new Attribute("Pixel"+(i+1)));
        }
        for (int i = 0; i < 7; i++)
        {
        	 attVals.add(""+i);
        }
        Attribute Class1 = new Attribute("Class", attVals);
        atts.add(Class1);
        data = new Instances("MyRelation", atts, 0);
        for(int j=0; j<Class.length;j++)
        {
        vals = new double[data.numAttributes()];
        for(int i=0;i<AttSize;i++)
        {
        	vals[i] = Attribute[j][i];
        }
        vals[AttSize] = attVals.indexOf(Integer.toString(Class[j]));
        
        
        Instance inst = new DenseInstance(1.0, vals);
        data.add(inst);
        }
        //System.out.println(data);
        Random rand = new Random();
        data.randomize(rand);
        // save data to arff File
        ArffSaver saver = new ArffSaver();
        saver.setInstances(data);
        saver.setFile(new File("fer2017Random.arff"));
        saver.writeBatch();
        
	}

}
