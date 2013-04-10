import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.TreeMap;

import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class KnnWithWeights {
	private int k = 5;
	private double[] attributesMax;;
	private double[] attributesMin;
	private Instances trainingSet;
	private Instances testSet;
	private double[] weights;
	private double[][] normalizedTraining;
	private double[][] normalizedTest;

	public KnnWithWeights(int k) {
		this.k = k;
	}
	
	public void buildClassifier(Instances trainingSet) {
		// We should check the attributes first
		this.trainingSet = trainingSet;
		getNumRange();
		normalizedTraining = normalize(trainingSet);

//		for (double[] a : normalizedTraining) {
//			System.out.println(Arrays.toString(a));
//		}		
		weights = new double[trainingSet.numAttributes()];
		
		for (int i = 0; i < weights.length; i++) {
			weights[i] = 1;
		}
		
		
	}
	
	public ArrayList<Double> classifyWithWeights(Instances trainSet, Instances testSet) {
		this.trainingSet = trainSet;
		this.testSet = testSet;
		getNumRange();
		ArrayList<Double> r = new ArrayList<Double>();
		normalizedTraining = normalize(trainingSet);
		normalizedTest = normalize(testSet);
		
 		for (int i = 0; i < normalizedTest.length; i++) {			
			TreeMap<Double, Integer> map = new TreeMap<Double, Integer>();
//			System.out.println("Test Instance: " + testSet.instance(0));
//			System.out.println(Arrays.toString(normalizedTest[0]));
//			System.out.println();
			for (int j = 0; j < trainingSet.numInstances(); j++) {			
				double similarity = getSimilarity(i, j);
//				System.out.println("Training instance " + j + ": " + trainingSet.instance(j));
//				System.out.println("Normalized: " + Arrays.toString(normalizedTraining[j]));
//				System.out.println("similarity: " + similarity);
				map.put(similarity, j);	
				//System.out.println(map.get(similarity));
			}
//			
//			while (map.size() > 181) {
//				Double key = map.lastKey();
//				System.out.println("key: " + key);
//				int index = map.remove(key);
//				Instance in = trainingSet.instance(index);
//				System.out.println("Instance: " + in);	
//				System.out.println(Arrays.toString(normalizedTraining[i ndex]));
//			}
						
			double[] values = new double[trainingSet.classAttribute().numValues()];
			for (int j = 0; j < k; j++) {				
				double key = map.lastKey();
				int index = map.remove(key);
//				System.out.println("Instance No" + index);
//				System.out.println(trainingSet.instance(index));
//				System.out.println(key);
				values[(int) trainingSet.instance(index).classValue()] += key;
			}
			
//			System.out.println(Arrays.toString(values));
//			System.out.println(index);
			r.add((double) getMaxIndex(values));
			
			System.out.println(Arrays.toString(values));
		}
	
		System.out.println(Arrays.toString(r.toArray()));
		
		return r;
	}

	private int getMaxIndex(double[] a) {
		int maxIndex = 0;
		for (int i = 0; i < a.length; i++) {
			if (a[i] > a[maxIndex]) {
				maxIndex = i;
			}
		}
		
		return maxIndex;
	}
	
	private double getSimilarity(int testIndex, int trainingIndex) {
		double r = 0;
		
		for (int i = 0; i < trainingSet.numAttributes(); i++) {
			if (i == trainingSet.classIndex()) continue;
			else if (trainingSet.attribute(i).isNumeric()) {
				r += euclideanNorm(normalizedTraining[trainingIndex][i], normalizedTest[testIndex][i]);
			} else if (trainingSet.attribute(i).isNominal() 
					    && normalizedTraining[trainingIndex][i] != normalizedTest[testIndex][i]) {
				r += 1;
			}
		}
		
		if (r == 0) return Double.MAX_VALUE;
		return 1 / (Math.sqrt(r));
	}
	
	private double normalizeHelper(double x, double xMax, double xMin) {
		double r = Math.pow((x - xMin) / (xMax - xMin), 2);
		return Math.sqrt(r);
	}

	private double euclideanNorm(double x, double y) {
		return Math.pow((x - y), 2);
	}
	
	private void getNumRange() {
		attributesMax = new double[trainingSet.numAttributes()];
		attributesMin = new double[trainingSet.numAttributes()];
				
		for (int i = 0; i < trainingSet.numInstances(); i++) {
			Instance in = trainingSet.get(i);			
			for (int j = 0; j < trainingSet.numAttributes(); j++) {
				if (j == trainingSet.classIndex()) {
					continue;
				}
											
				if (in.attribute(j).isNumeric()) {	
					double d = in.value(j);	
					if (i == 0) {
						attributesMax[j] = d;
						attributesMin[j] = d;	
						continue;
					} else if (attributesMax[j] < d) {
						attributesMax[j] = d;
					} else if (attributesMin[j] > d) {
						attributesMin[j] = d;
					}
				} 
			}
		}
		
		for (int i = 0; i < testSet.numInstances(); i++) {
			Instance in = testSet.get(i);			
			for (int j = 0; j < testSet.numAttributes(); j++) {
				if (j == testSet.classIndex()) {
					continue;
				}
											
				if (in.attribute(j).isNumeric()) {	
					double d = in.value(j);	
					if (attributesMax[j] < d) {
						attributesMax[j] = d;
					} else if (attributesMin[j] > d) {
						attributesMin[j] = d;
					}
				} 
			}
		}
		
//		System.out.println(Arrays.toString(this.attributesMax));
//		System.out.println(Arrays.toString(this.attributesMin));
	}
	
	private double[][] normalize(Instances data) {
		double[][] normalized = new double[data.numInstances()][data.numAttributes()];
		
		for (int i = 0; i < normalized.length; i++) {
			Instance in = data.instance(i);
			for (int j = 0; j < normalized[i].length - 1; j++) {
				if (j == data.classIndex()) {
					normalized[i][j] = in.classValue();
				} else if (in.attribute(j).isNumeric()) {
					normalized[i][j] = normalizeHelper(in.value(j), attributesMax[j], attributesMin[j]);
				} else if (in.attribute(j).isNominal()) {
					normalized[i][j] = in.value(j);
				} 				

			}
		}
		
		return normalized;
	}

	public static void main(String[] args) {
		DataSource source1 = null;
		DataSource source2 = null;
		try {
			source1 = new DataSource(
					"trainProdSelection.arff");
			source2 = new DataSource(
					"testProdSelection.arff");
			KnnWithWeights knn = new KnnWithWeights(3);
			Instances inst1 = source1.getDataSet();
			Instances inst2 = source2.getDataSet();
//			List<Double> list = knn.clasifyInstances(source1.getDataSet(), source2.getDataSet());
			inst1.setClassIndex(source1.getDataSet().numAttributes()-1);
			inst2.setClassIndex(source2.getDataSet().numAttributes()-1);
			
			knn.classifyWithWeights(inst1, inst2);
//			for(int i=0;i<list.size();i++) {
//				System.out.println(inst1.classAttribute().value(list.get(i).intValue()));
//			}
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
}
