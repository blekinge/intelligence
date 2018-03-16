package dk.kb.tensorflow;

import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;

import org.tensorflow.Graph;
import org.tensorflow.Operation;
import org.tensorflow.Output;
import org.tensorflow.Session;
import org.tensorflow.Tensor;

/**
 * 
 * Using https://github.com/tensorflow/tensorflow/blob/r1.6/tensorflow/java/src/main/java/org/tensorflow/examples/LabelImage.java
 * as example for using TensorFlow in java
 *
 */
public class MyLabelImage {

    public static void main(String[] args) {
        if (args.length < 2) {
            System.err.println("Missing args. Needs 2 arguments (modelDir, imagefile). Only given " +  args.length);
            System.exit(1);       
        }
        String modelDir = args[0];
        String imageFile = args[1];
        
        byte[] imageBytes = readAllBytesOrExit(Paths.get(imageFile));
        byte[] graphDef = readAllBytesOrExit(Paths.get(modelDir, "tensorflow_inception_graph.pb"));
        List<String> labels = readAllLinesOrExit(Paths.get(modelDir, "imagenet_comp_graph_label_strings.txt"));
        Tensor<Float> image = constructAndExecuteGraphToNormalizeImage(imageBytes);
        
        float[] labelProbabilities = executeInceptionGraph(graphDef, image);
        int bestLabelIdx = maxIndex(labelProbabilities);
        System.out.println(
                String.format("BEST MATCH: %s (%.2f%% likely)",
                    labels.get(bestLabelIdx),
                    labelProbabilities[bestLabelIdx] * 100f));

    }
    
    private static byte[] readAllBytesOrExit(Path path) {
        System.out.println(path);
        try {
          return Files.readAllBytes(path);
        } catch (IOException e) {
          System.err.println("Failed to read [" + path + "]: " + e.getMessage());
          System.exit(1);
        }
        return null;
      }
    
    private static List<String> readAllLinesOrExit(Path path) {
        try {
          return Files.readAllLines(path, Charset.forName("UTF-8"));
        } catch (IOException e) {
          System.err.println("Failed to read [" + path + "]: " + e.getMessage());
          System.exit(0);
        }
        return null;
    }
    
    private static Tensor<Float> constructAndExecuteGraphToNormalizeImage(byte[] imageBytes) {
        try (Graph g = new Graph()) {
          GraphBuilder b = new GraphBuilder(g);
          // Some constants specific to the pre-trained model at:
          // https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip
          //
          // - The model was trained with images scaled to 224x224 pixels.
          // - The colors, represented as R, G, B in 1-byte each were converted to
          //   float using (value - Mean)/Scale.
          final int H = 224;
          final int W = 224;
          final float mean = 117f;
          final float scale = 1f;

          // Since the graph is being constructed once per execution here, we can use a constant for the
          // input image. If the graph were to be re-used for multiple input images, a placeholder would
          // have been more appropriate.
          final Output<String> input = b.constant("input", imageBytes);
          final Output<Float> output =
              b.div(
                  b.sub(
                      b.resizeBilinear(
                          b.expandDims(
                              b.cast(b.decodeJpeg(input, 3), Float.class),
                              b.constant("make_batch", 0)),
                          b.constant("size", new int[] {H, W})),
                      b.constant("mean", mean)),
                  b.constant("scale", scale));
          try (Session s = new Session(g)) {
            System.out.println("OP: " +   output.op().name());
            return s.runner()
                    .fetch(output.op().name())
                    .run()
                    .get(0)
                    .expect(Float.class);
          }
        }
    }
    
    private static float[] executeInceptionGraph(byte[] graphDef, Tensor<Float> image) {
        try (Graph g = new Graph()) {
          g.importGraphDef(graphDef);
          describeGraph(g);
          
          try (Session s = new Session(g);
              Tensor<Float> result =
                  s.runner().feed("input", image).fetch("output").run().get(0).expect(Float.class)) {
            final long[] rshape = result.shape();
            if (result.numDimensions() != 2 || rshape[0] != 1) {
              throw new RuntimeException(
                  String.format(
                      "Expected model to produce a [1 N] shaped tensor where N is the number of labels, instead it produced one with shape %s",
                      Arrays.toString(rshape)));
            }
            int nlabels = (int) rshape[1];
            return result.copyTo(new float[1][nlabels])[0];
          }
        }
      }

      private static void describeGraph(Graph g) {
          System.out.println(g);
          Iterator<Operation> i = g.operations();
          int j=0;
          while(i.hasNext()) {
              System.out.println("OPS #" + ++j + ": " + i.next().name());
          }
        
    }

    private static int maxIndex(float[] probabilities) {
        int best = 0;
        for (int i = 1; i < probabilities.length; ++i) {
          if (probabilities[i] > probabilities[best]) {
            best = i;
          }
        }
        return best;
      }

    
    
    
}
