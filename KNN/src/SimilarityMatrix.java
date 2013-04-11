/*
 * Team 13 Prodigy
 * SimilarityMatrix used to read in similarities between nominal attributes.
 */

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Scanner;

public class SimilarityMatrix {
	private String[] index;
	private double[][] matrix;
	private int count_index;

	public SimilarityMatrix (String filename) {
		Scanner scanner = null;
		File file = new File(filename);
		try {
			scanner = new Scanner(file);
			if (scanner.hasNextLine()) {
				String line = scanner.nextLine();

				String[] temp = line.split("\\W");
				String[] temp2 = new String[temp.length - 1];
				for (String s : temp) {
					if (!s.matches("[a-zA-Z]+$")) {
						temp2[count_index] = s;
						count_index++;
					}
				}
				index = temp2;

			}
			ArrayList<double[]> tempList = new ArrayList<double[]>();
			while (scanner.hasNextLine()) {
				String line = scanner.nextLine();
				
				String[] temp = line.split(",");
				double[] temp2 = new double[temp.length - 1];
				int i = 0;
				for (String s : temp) {
					if (!s.equals("") && s.matches("\\d*.?\\d*+")) {
						temp2[i] = Double.parseDouble(s);
						i++;
					}
				}
				tempList.add(temp2);
			}
			matrix = new double[tempList.size()][tempList.get(0).length];
			for(int i=0; i<tempList.size(); i++){
				matrix[i] = tempList.get(i);
			}
			// for(String s:index)
			// System.out.println(s);

		} catch (FileNotFoundException e) {
			System.err.println("Cannot find the file: " + filename);
			return;
		} finally {
			if (scanner != null)
				scanner.close();
		}
		
 
	}
	public double getSimilarity(int valIndex1, int valIndex2){
		return matrix[valIndex2][valIndex1];
	}
	public static void main(String[] args) {
		SimilarityMatrix s = new SimilarityMatrix("type.txt");
		double ds = s.getSimilarity(3, 3);
		System.out.print(ds);
	}

}
