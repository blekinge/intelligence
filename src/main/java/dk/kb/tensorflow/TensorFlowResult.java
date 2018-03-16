package dk.kb.tensorflow;

import java.util.List;

public class TensorFlowResult {
    private String label;
    private double accuracy;
    private String objectName;

    public TensorFlowResult(String label, double accuracy, String objectName) {
        this.label = label;
        this.accuracy = accuracy;
        this.objectName = objectName;
    }
 
    public String getLabel() {
        return label;
    }

    public double getAccuracy() {
        return accuracy;
    }
    
    public static TensorFlowResult getResult(List<String> labelList, float[] output, String objectName) {
        float max = 0L;
        int maxIndex = -1;
        for (int i=0; i < output.length; i++) {
            if (output[i] > max) {
                max = output[i];
                maxIndex = i;
            }
        }
        return new TensorFlowResult(labelList.get(maxIndex), max, objectName);
    }

    public String getObjectName() {
        return objectName;
    }
    
    public String toString() {
        return "The object '" + objectName + "' is classified as '" +  label + "' with accuracy '" +  accuracy + "'";
    }
    
    
}
