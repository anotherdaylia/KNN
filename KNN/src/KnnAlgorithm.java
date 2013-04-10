import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class KnnAlgorithm {
	private int k = 5;
	private String[] attributes;
	private double[] attributesMax;;
	private double[] attributesMin;
	private String[] attributeType;
	private Instances trainingSet;
	private Instances testSet;

	public KnnAlgorithm(int k) {
		this.k = k;
	}

	public List<Double> clasifyInstances(Instances trainingSet, Instances testSet) {
		try {
			this.trainingSet = trainingSet;
			this.testSet = testSet;
			this.trainingSet.setClassIndex(trainingSet.numAttributes()-1);
			this.testSet.setClassIndex(testSet.numAttributes()-1);
			attributes = new String[trainingSet.numAttributes()];
			attributesMax = new double[trainingSet.numAttributes()];
			attributesMin = new double[trainingSet.numAttributes()];
			attributeType = new String[trainingSet.numAttributes()];
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

			// String[] nominalValues = new String[data.numAttributes()];
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
			List<Double> result = new ArrayList<Double>();
			for (int i = 0; i < testDataValues.length; i++) {
				Map<Double, Double> map = new TreeMap<Double, Double>();
				for (int j = 0; j < trainingDataValues.length; j++) {
					double[] attributesValues = new double[trainingDataValues[0].length];
					double sum = 0.0;
					for (int k = 0; k < attributesValues.length - 1; k++) {
						if (attributeType[k].equals("real")) {
							attributesValues[k] = euclideanNorm(
									testDataValues[i][k],
									trainingDataValues[j][k]);
						} else {
							if(trainingSet.get(j).value(k) == testSet.get(i).value(k)) {
								attributesValues[k] = 1;
							} else {
								attributesValues[k] = 0;
							}
						}
						sum = sum + attributesValues[k];
					}
					double distance = 1 / (Math.sqrt(sum));
					map.put(distance, trainingSet.get(j).classValue());
				}
				List<Double> list = new ArrayList<Double>(map.values());
				int[] arr = new int[k];
				double tmp = 0.0;
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
				
				result.add(tmp);
			}
			return result;
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
			source1 = new DataSource(
					"trainProdSelection.arff");
			source2 = new DataSource(
					"testProdSelection.arff");
			KnnAlgorithm knn = new KnnAlgorithm(3);
			Instances inst1 = source1.getDataSet();
			Instances inst2 = source1.getDataSet();
			List<Double> list = knn.clasifyInstances(source1.getDataSet(), source2.getDataSet());
			inst1.setClassIndex(source1.getDataSet().numAttributes()-1);
			inst2.setClassIndex(source2.getDataSet().numAttributes()-1);
			for(int i=0;i<list.size();i++) {
				System.out.println(inst1.classAttribute().value(list.get(i).intValue()));
			}
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
}