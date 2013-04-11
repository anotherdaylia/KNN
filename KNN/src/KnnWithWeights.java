/*
 * Team 13 Prodigy
 * A simple classifier that implements KNN algorithm.
 * It calculates the similarities between nominal attributes given the similarity matrixes from text files.
 * It calculates the similarities between numeric attributes based on euclidean methods.
 * It assigns weights to attributes through a learning process.
 */

import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
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
	HashMap<String, SimilarityMatrix> matrixes;

	public KnnWithWeights(int k) {
		this.k = k;
		matrixes = new HashMap<String, SimilarityMatrix>();
	}
	
	// Given the training set and test set, return the predicted results.
	// Return the index of the attribute values if the class attribute is nominal
	// Return the numeric values if the class attribute is numeric
	public ArrayList<Double> classifyInstances(Instances trainingSet, Instances testSet) {
		for (int i = 0; i < trainingSet.numAttributes(); i++) {
			if (i != trainingSet.classIndex() && trainingSet.attribute(i).isNominal()) {
				String name = trainingSet.attribute(i).name();
				SimilarityMatrix m = new SimilarityMatrix(name + ".txt");
				matrixes.put(name, m);
			}
		}
		
		setup(trainingSet, testSet);
		buildClassifier();
		setup(trainingSet, testSet);
		ArrayList<Double> r = new ArrayList<Double>();
		
 		for (int i = 0; i < normalizedTest.length; i++) {			
			TreeMap<Double, Integer> map = new TreeMap<Double, Integer>();
			for (int j = 0; j < trainingSet.numInstances(); j++) {	
				double similarity = getSimilarity(i, j);
				map.put(similarity, j);	
			}				
			r.add(getClassValue(map));
		}		
		return r;
	}
	
	private void setup(Instances trainingSet, Instances testSet) {
		this.trainingSet = trainingSet;
		this.testSet = testSet;
		getNumRange();
		normalizedTraining = normalize(trainingSet);
		normalizedTest = normalize(testSet);
	}
	
	// Build the weights for KNN
	public void buildClassifier() {
		weights = new double[trainingSet.numAttributes()];
		
		for (int i = 0; i < weights.length; i++) {
			weights[i] = 0.01;
		}
		
		CrossValidation cv = new CrossValidation(trainingSet, 5);
		boolean change = true;
		while (change) {
			change = false;
			for (int i = 0; i < trainingSet.numAttributes(); i++) {
				if (i == trainingSet.classIndex()) continue;
				double oldW = weights[i];
				testStep(cv, i, 0.5);
				testStep(cv, i, 0.1);	
				if (!change) {
					change = weights[i] - oldW > CrossValidation.epsilon;
				}
			}
		}
	}
	
	// Test the weights
	private void testStep(CrossValidation cv, int attrIndex, double step) {
		double oldAccuracy = 0;
		double newAccuracy = cv.weightsCrossvalidation(trainingSet, this, weights);
		while (true) {
			weights[attrIndex] += step;
			oldAccuracy = newAccuracy;
			newAccuracy = cv.weightsCrossvalidation(trainingSet, this, weights);
			
			if (oldAccuracy == newAccuracy || oldAccuracy - newAccuracy >= CrossValidation.epsilon) {
				weights[attrIndex] -= step;
				break;
			}
		}
	}
	
	// Helper method to evaluate the performance of weights
	public ArrayList<Double> classifyWithWeights(Instances trainingSet, Instances testSet, double[] weights) {
		ArrayList<Double> r = new ArrayList<Double>();
		this.trainingSet = trainingSet;
		this.testSet = testSet;
		this.weights = weights;
		getNumRange();
		normalizedTraining = normalize(trainingSet);
		normalizedTest = normalize(testSet);
		
 		for (int i = 0; i < normalizedTest.length; i++) {			
			TreeMap<Double, Integer> map = new TreeMap<Double, Integer>();
			for (int j = 0; j < trainingSet.numInstances(); j++) {			
				double similarity = getSimilarity(i, j);
				map.put(similarity, j);	
			}			
			r.add(getClassValue(map));
		}	
		return r;
	}
	
	private double getClassValue(TreeMap<Double, Integer> map) {
		if (trainingSet.classAttribute().isNominal()) {	
			double[] values = new double[trainingSet.classAttribute().numValues()];
			for (int j = 0; j < k; j++) {				
				double key = map.lastKey();
				int index = map.remove(key);
				values[(int) trainingSet.instance(index).classValue()] += key;
			}
			
			return getMaxIndex(values);
		} else {
			double r = 0;			
			double totalS = 0;
			for (int i = 0; i < k; i++) {
				double key = map.lastKey();
				totalS += key;
				int index = map.remove(key);
				r += trainingSet.instance(index).classValue() * key;
			}		
			return r / totalS;
		}	
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
				r += weights[i] * euclideanNorm(normalizedTraining[trainingIndex][i], normalizedTest[testIndex][i]);
			} else if (trainingSet.attribute(i).isNominal()) {
				SimilarityMatrix m = matrixes.get(trainingSet.attribute(i).name());
				r += weights[i] * (1 - m.getSimilarity((int) trainingSet.instance(trainingIndex).value(i), 
											(int) testSet.instance(testIndex).value(i)));
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
		
		if (testSet == null) return;
		
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
			inst1.setClassIndex(source1.getDataSet().numAttributes()-1);
			inst2.setClassIndex(source2.getDataSet().numAttributes()-1);
			
			List<Double> list = knn.classifyInstances(inst1, inst2);
			for(int i=0;i<list.size();i++) {
				System.out.println(list.get(i));
			}
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
}
