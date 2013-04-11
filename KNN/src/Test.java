/*
 * Team 13 Prodigy
 * Testing for KNN algorithm
 * Generating the reports results for task11
 */

import java.io.FileNotFoundException;
import java.util.List;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;


public class Test {
	public static void main(String[] args) {
		DataSource trainingSource = null;
		DataSource testSource = null;
		Instances trainingSet = null;
		Instances testSet = null;
		try {
			// Part a: product selection
			trainingSource = new DataSource(
					"trainProdSelection.arff");
			testSource = new DataSource(
					"testProdSelection.arff");
			KnnWithWeights knn = new KnnWithWeights(3);

			trainingSet = trainingSource.getDataSet();
			testSet = testSource.getDataSet();
			trainingSet.setClassIndex(trainingSet.numAttributes()-1);
			testSet.setClassIndex(testSet.numAttributes()-1);
			
			List<Double> list = knn.classifyInstances(trainingSet, testSet);
			System.out.println("Part b: production selection");
			for(int i=0;i<list.size();i++) {
				System.out.println(list.get(i));
			}
			
			// Part b: product introduction (binary)
			trainingSource = new DataSource(
					"trainProdIntro.binary.arff");
			testSource = new DataSource(
					"testProdIntro.binary.arff");
			knn = new KnnWithWeights(1);

			trainingSet = trainingSource.getDataSet();
			testSet = testSource.getDataSet();
			trainingSet.setClassIndex(trainingSet.numAttributes()-1);
			testSet.setClassIndex(testSet.numAttributes()-1);
			
			list = knn.classifyInstances(trainingSet, testSet);
			System.out.println("-----------------------------");			
			System.out.println("Part b: production introduction (binary)");
			for(int i=0;i<list.size();i++) {
				System.out.println(list.get(i));
			}
			
			// Part b: product introduction (real)
			trainingSource = new DataSource(
					"trainProdIntro.real.arff");
			testSource = new DataSource(
					"testProdIntro.real.arff");
			knn = new KnnWithWeights(1);

			trainingSet = trainingSource.getDataSet();
			testSet = testSource.getDataSet();
			trainingSet.setClassIndex(trainingSet.numAttributes()-1);
			testSet.setClassIndex(testSet.numAttributes()-1);
			
			list = knn.classifyInstances(trainingSet, testSet);
			System.out.println("-----------------------------");			
			System.out.println("Part a: production introduction (real)");
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
