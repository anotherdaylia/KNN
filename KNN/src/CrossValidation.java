import java.util.ArrayList;
import java.util.Random;

import weka.classifiers.trees.J48;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class CrossValidation {
	private Instances dataSet; 
	private int k;           //folds
	private int sizeOfInput; //size of entire data size
	private ArrayList<Instances> bigArrayList;
	
	public static void main(String[] args) {
		DataSource source = null;
		Instances data = null;
		try {
			source = new DataSource("whether.arff");
			data = source.getDataSet();
			if (data.classIndex() == -1)
				data.setClassIndex(data.numAttributes() - 1);
			J48 j48 = new J48();
			j48.buildClassifier(data);
			CrossValidation cv = new CrossValidation(data, 5);
			cv.doCrossValidation(data, j48);
		} catch (Exception e) {
			e.printStackTrace();
		}
		
		
	}
	
	public CrossValidation(Instances dataSet, int k){
		this.dataSet = dataSet;
		this.sizeOfInput = dataSet.size();
		this.k = k;
		this.bigArrayList = generateFolds();
	}
	
	private int[] performPermutation() {
        int j = 0;
        int k; 
        Random rand = new Random();
        
        int[] dataNum = new int[sizeOfInput];
        for (int i = 0; i < sizeOfInput; i++) {
                dataNum[i] = i;
        }

        for (int i = sizeOfInput - 1; i > 0; i--) {
                j = rand.nextInt(i+1); // random integer between 0 and i
                k = dataNum[i];
                dataNum[i] = dataNum[j];
                dataNum[j] = k;
        }
        return dataNum;
	}
	
	public ArrayList<Instances> generateFolds(){
		bigArrayList = new ArrayList<Instances>();
        int[] dataNum = new int[sizeOfInput];
        dataNum = performPermutation();
        int sizePerFold = sizeOfInput/k;
        
        // use bigArrayList to store k folds
        for (int i=0; i<k; i++){
        	bigArrayList.add(new Instances(dataSet, sizeOfInput));
        }
        
        int count=0;
        for(int i=0; i<k; i++){
        	for(int j=count; j<sizePerFold; j++){
        		bigArrayList.get(i).add(dataSet.get(dataNum[j]));
        	}
        	count += sizePerFold;
        }

        return bigArrayList;
	}
	
	public Instances getTrainingData(int n){
		int sizePerFold = sizeOfInput/k;
		int sizeOfTrainingData = sizeOfInput - sizePerFold;
		
		Instances trainingData = new Instances(dataSet, sizeOfTrainingData);
		
		for(int i=0; i<k; i++){
			if(i == n) continue;
			Instances oneFold = bigArrayList.get(i);
			trainingData.addAll(oneFold);
//			for(int j=0; i<sizePerFold; j++){
//				trainingData.add(oneFold.get(j));
//			}
		}
		return trainingData;
	}
	
	public double doCrossValidation(Instances dataSet, KNN knn){
		double accuracy = 0.0;
		double[] accuracyPerFold = new double[k];
		int isTheSame = 0;
		int sizePerFold = sizeOfInput/k;
		
		for(int i=0; i<k; i++){
			Instances oneFold = bigArrayList.get(i);
			Instances predictList = knn.classifyAttribute(getTrainingData(i), bigArrayList.get(i));
			
			for(int j=0; j<sizePerFold; j++){
				if(predictList.get(j).equals(oneFold.get(j))){
					isTheSame++;
				}
			}
			accuracyPerFold[i] = (double)isTheSame/(double)sizePerFold;
		}
		
		for(int i=0; i<k; i++){
			accuracy += accuracyPerFold[i];
		}
		accuracy = accuracy/k;
		return accuracy;
	}
	
	public double doCrossValidation(Instances dataSet, J48 knn) throws Exception{
		double accuracy = 0.0;
		double[] accuracyPerFold = new double[k];
		int isTheSame = 0;
		int sizePerFold = sizeOfInput/k;
		
		
		
		for(int i=0; i<k; i++){
			Instances test = bigArrayList.get(i);
			
			//System.out.println(test);
			
			for (int j = 0; j < sizePerFold; j++) {
				Instance in = test.get(j);
//				System.out.println(knn);
//				System.out.println(in);
				if(knn.classifyInstance(in) == in.classValue()) {
					isTheSame++;
				}
			}
			
			accuracyPerFold[i] = (double)isTheSame/(double)sizePerFold;
		}
		
		for(int i=0; i<k; i++){
			accuracy += accuracyPerFold[i];
		}
		accuracy = accuracy/k;
		return accuracy;
	}
	
	public class KNN {
		public Instances classifyAttribute(Instances trainingData, Instances testData){
			return new Instances(testData, testData.size());
		}	
	}
	
}
