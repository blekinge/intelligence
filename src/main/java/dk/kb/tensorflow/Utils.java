package dk.kb.tensorflow;

import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Iterator;
import java.util.List;

import org.tensorflow.Graph;
import org.tensorflow.Operation;

public class Utils {
    public static byte[] readAllBytesOrExit(Path path) {
        try {
          return Files.readAllBytes(path);
        } catch (IOException e) {
          System.err.println("Failed to read [" + path + "]: " + e.getMessage());
          System.exit(1);
        }
        return null;
      }
    
    public static List<String> readAllLinesOrExit(Path path) {
        try {
          return Files.readAllLines(path, Charset.forName("UTF-8"));
        } catch (IOException e) {
          System.err.println("Failed to read [" + path + "]: " + e.getMessage());
          System.exit(0);
        }
        return null;
    }
    
    public static void describeGraph(Graph g) {
        System.out.println(g);
        Iterator<Operation> i = g.operations();
        int j=0;
        while(i.hasNext()) {
            System.out.println("OPS #" + ++j + ": " + i.next().name());
        }
      
  }

  public static int maxIndex(float[] probabilities) {
      int best = 0;
      for (int i = 1; i < probabilities.length; ++i) {
        if (probabilities[i] > probabilities[best]) {
          best = i;
        }
      }
      return best;
    }
}
