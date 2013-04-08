import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

import weka.core.Instances;
import weka.core.converters.ArffLoader.ArffReader;
import weka.core.converters.ConverterUtils.DataSource;

public class knnAlgorithm {
	private final int k = 90;
	private String[] attributes;
	private double[] attributesMax;;
	private double[] attributesMin;
	private String[] attributeType;
	private Instances trainingSet;
	private Instances testSet;

	public knnAlgorithm(Instances trainingSet, Instances testSet) {
		this.trainingSet = trainingSet;
		this.testSet = testSet;
		attributes = new String[trainingSet.numAttributes()];
		attributesMax = new double[trainingSet.numAttributes()];
		attributesMin = new double[trainingSet.numAttributes()];
		attributeType = new String[trainingSet.numAttributes()];
	}

	public String storeData() {
		try {

			// String[] nominalValues = new String[data.numAttributes()];
			for (int i = 0; i < trainingSet.numAttributes(); i++) {
				attributes[i] = trainingSet.attribute(i).toString();
				if (attributes[i].contains("real")
						|| attributes[i].contains("numeric")) {
					attributeType[i] = "real";
				} else {
					attributeType[i] = "nominal";
				}
				attributesMax[i] = 0;
				attributesMin[i] = 0;
			}
			Double[][] trainingDataValues = new Double[trainingSet
					.numInstances()][trainingSet.numAttributes()];
			Double[][] testDataValues = new Double[testSet.numInstances()][testSet
					.numAttributes()];
			String[] predictionValues = new String[trainingSet.numInstances()];
			for (int i = 0; i < trainingSet.numInstances(); i++) {
				String[] str = new String[trainingSet.numAttributes()];
				str = trainingSet.instance(i).toString().split(",");
				for (int j = 0; j < trainingSet.numAttributes(); j++) {
					if (j == trainingSet.numAttributes() - 1) {
						predictionValues[i] = str[j];
						break;
					}
					if (attributeType[j].equals("real")) {
						double d = Double.parseDouble(str[j]);
						if (attributesMax[j] < d) {
							attributesMax[j] = d;
						}
						if (attributesMin[j] > d) {
							attributesMin[j] = d;
						}
						trainingDataValues[i][j] = d;
					} else {
						trainingDataValues[i][j] = 0.0;
						attributesMax[j] = 1.0;
						attributesMin[j] = 0.0;
					}
				}
			}
			normalize(trainingDataValues, attributesMax, attributesMin);
			for (int i = 0; i < testSet.numInstances(); i++) {
				String[] str = new String[testSet.numAttributes()];
				str = trainingSet.instance(i).toString().split(",");
				for (int j = 0; j < trainingSet.numAttributes() - 1; j++) {

					if (attributeType[j].equals("real")) {
						double d = Double.parseDouble(str[j]);
						if (attributesMax[j] < d) {
							attributesMax[j] = d;
						}
						if (attributesMin[j] > d) {
							attributesMin[j] = d;
						}
						testDataValues[i][j] = d;
					} else {
						testDataValues[i][j] = 0.0;
						attributesMax[j] = 1.0;
						attributesMin[j] = 0.0;
					}
				}
			}

			normalize(testDataValues, attributesMax, attributesMin);
			StringBuilder sb = new StringBuilder();
			for (int i = 0; i < testDataValues.length; i++) {
				Map<Double, String> map = new TreeMap<Double, String>();
				for (int j = 0; j < trainingDataValues.length; j++) {
					double[] attributesValues = new double[trainingDataValues[0].length];
					double sum = 0.0;
					for (int k = 0; k < attributesValues.length - 1; k++) {
						attributesValues[k] = euclideanNorm(
								testDataValues[i][k], trainingDataValues[j][k]);
						sum = sum + attributesValues[k];
					}
					double distance = 1 / (Math.sqrt(sum));
					map.put(distance, predictionValues[j]);
				}
				List<String> list = new ArrayList<String>(map.values());
				int[] arr = new int[k];
				String tmp = "";
				int max = 0;
				for (int p = 0; p < k; p++) {
					for (int q = 0; q < k; q++) {
						if (list.get(p).equals(list.get(q))) {
							arr[p]++;
						}
					}
					if (max < arr[p]) {
						max = arr[p];
						tmp = list.get(p);
					}

				}
				sb.append(tmp);
				sb.append(",");
			}
			sb.replace(sb.length() - 1, sb.length(), "");
			return sb.toString();
		} catch (Exception e) {
			e.printStackTrace();
			return null;
		}

	}

	private void normalize(Double[][] dataValues, double[] attributesMax,
			double[] attributesMin) {
		for (int i = 0; i < dataValues.length; i++) {
			for (int j = 0; j < dataValues[j].length - 1; j++) {
				dataValues[i][j] = Math.sqrt(normalizeHelper(dataValues[i][j],
						attributesMax[j], attributesMin[j]));
			}
		}
	}

	private static double normalizeHelper(double x, double xMax, double xMin) {
		return Math.pow((x - xMin) / (xMax - xMin), 2);
	}

	private static double euclideanNorm(double x, double y) {
		return Math.pow((x - y), 2);
	}

	public static void main(String[] args) {
		DataSource source1 = null;
		DataSource source2 = null;
		try {
			source1 = new DataSource("C:/Users/vivek/Downloads/task11a_2013(3)/task11a_2013/attachments/trainProdSelection/trainProdSelection.arff");
			source2 = new DataSource("C:/Users/vivek/Downloads/task11a_2013(3)/task11a_2013/attachments/testProdSelection/testProdSelection.arff");
			knnAlgorithm knn = new knnAlgorithm(source1.getDataSet(), source2.getDataSet());
			System.out.println(knn.storeData());
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
}
