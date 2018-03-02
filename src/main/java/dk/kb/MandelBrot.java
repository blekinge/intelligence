package dk.kb;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

import javax.imageio.ImageIO;

import org.tensorflow.Graph;
import org.tensorflow.Output;
import org.tensorflow.Session;
import org.tensorflow.Tensor;

/**
 * An attempt to re-implement the python code on https://www.tensorflow.org/tutorials/mandelbrot
 * in java
 * Using https://github.com/tensorflow/tensorflow/blob/r1.6/tensorflow/java/src/main/java/org/tensorflow/examples/LabelImage.java
 * as example
 *
 */
public class MandelBrot {

    public static void main(String[] args) {
        String path = "/home/svc/devel/intelligence/strawberry.jpg";
        File pict = new File(path);
        BufferedImage img = null;
        try {
            img = ImageIO.read(pict);
        } catch (IOException e) {
        }
        byte[] imageBytes = readAllBytesOrExit(Paths.get(path));
        Tensor<Float> image = constructAndExecuteGraphToNormalizeImage(imageBytes);
    }
    
    private static byte[] readAllBytesOrExit(Path path) {
        try {
          return Files.readAllBytes(path);
        } catch (IOException e) {
          System.err.println("Failed to read [" + path + "]: " + e.getMessage());
          System.exit(1);
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
            return s.runner().fetch(output.op().name()).run().get(0).expect(Float.class);
          }
        }
    }
}
