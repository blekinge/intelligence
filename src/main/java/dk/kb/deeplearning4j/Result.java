package dk.kb.deeplearning4j;

import org.nd4j.linalg.api.ndarray.INDArray;

public class Result {
    private int labelIndex;
    private double accuracy;
    private String objectName;

    public Result(int labelindex, double accuracy, String objectName) {
        this.labelIndex = labelindex;
        this.accuracy = accuracy;
        this.objectName = objectName;
    }
 
    public int getLabelIndex() {
        return labelIndex;
    }

    public double getAccuracy() {
        return accuracy;
    }
    
    public static Result getResult(INDArray output, String objectName) {
        double[] resultArray = output.data().asDouble();
        double max = 0L;
        int maxIndex = -1;
        for (int i=0; i < resultArray.length; i++) {
            if (resultArray[i] > max) {
                max = resultArray[i];
                maxIndex = i;
            }
        }
        return new Result(maxIndex, max, objectName);
    }

    public String getObjectName() {
        return objectName;
    }

    public void setObjectName(String objectName) {
        this.objectName = objectName;
    }
    
    
}
