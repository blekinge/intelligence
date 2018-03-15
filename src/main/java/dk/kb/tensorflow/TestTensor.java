package dk.kb.tensorflow;


import java.io.UnsupportedEncodingException;

import org.tensorflow.Tensor;
import org.tensorflow.TensorFlow;
import org.tensorflow.Tensors;

public class TestTensor {

    public static void main(String[] args) throws UnsupportedEncodingException {
       String value = "Hello from " + TensorFlow.version(); 
       byte[] data = value.getBytes("UTF-8"); 
       Tensor<String> t = Tensors.create(data);
       System.out.println(t.numDimensions());
       int[ ] num[ ] = {{0,2}, {1,2}, {2,2}, {3,2}, {4,23}};
       Tensor<Integer> it = Tensors.create(num);
       System.out.println(it.numDimensions());
       System.out.println(it.shape()[0]);
    }

}
