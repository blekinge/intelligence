package dk.kb.tensorflow;

import java.io.File;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.tensorflow.Graph;
import org.tensorflow.Output;
import org.tensorflow.Session;
import org.tensorflow.Tensor;

/**
 * 
 * Using https://github.com/tensorflow/tensorflow/blob/r1.6/tensorflow/java/src/main/java/org/tensorflow/examples/LabelImage.java
 * as example for using TensorFlow in java.
 * Tries to evaluate a picture as one of several pre-trained categories.
 * tested on front/KE044976.jpg, back/KE042847.jpg from corpus selected by DGJ
 * Assumes the following files are in the modelDir: graph.pb, labels.txt
 * 
 */
public class FrontBackClassifier {

    private static final String graphPbName = "graph.pb";
    private static final String labelsName = "labels.txt"; 
    private static final String OutputOperationName = "final_result";
    
    public static void main(String[] args) {
        if (args.length < 2) {
            System.err.println("Missing args. Needs 3 arguments (modelDir, imagefile). Only given " +  args.length);
            System.exit(1);       
        }
        String modelDirPath = args[0];
        String imageFilePath = args[1];
        File modelDir = new File(modelDirPath);
        if (!modelDir.exists()) {
            System.err.println("Did not found the given modeldir: " + modelDir.getAbsolutePath());
            System.exit(1);
        } else if (false == new File(modelDir, graphPbName).exists()) {
            System.err.println("Did not found the required file '" +  graphPbName + "' in the modeldir: " +  modelDir.getAbsolutePath());
            System.exit(1);
        } else if (false == new File(modelDir, labelsName).exists()) {
            System.err.println("Did not found the required file '" +  labelsName + "' in the modeldir: " +  modelDir.getAbsolutePath());
            System.exit(1);
        }
        File imageFileOrDir = new File(imageFilePath);
        
        FrontBackClassifier fbc = new FrontBackClassifier(modelDir);
        if (imageFileOrDir.isFile()) {
            TensorFlowResult r = fbc.evaluate(imageFileOrDir);
            System.out.println(r);
        } else {
            List<TensorFlowResult> results = fbc.evaluateDir(imageFileOrDir);
            System.out.println("Evaluated " + results.size() + " files ");
            for (TensorFlowResult r: results) {
                System.out.println(r);
            }
        }
    }
    
    private byte[] graphDef; // the pb file as a byte array
    private List<String> labels; // the labels file as a list of String
    private Graph execGraph; // The execution graph
    
    public FrontBackClassifier(File modelDir) {
        Path graphPbPath = Paths.get(modelDir.getAbsolutePath(), graphPbName);
        Path graphLabelPath = Paths.get(modelDir.getAbsolutePath(), labelsName);
        this.graphDef = Utils.readAllBytesOrExit(graphPbPath);
        this.labels = Utils.readAllLinesOrExit(graphLabelPath);
        this.execGraph = new Graph();
        this.execGraph.importGraphDef(graphDef);
    }
    
    public TensorFlowResult evaluate(File picture) {
        if (!picture.isFile()) {
            System.err.println("Ignoring file '" + picture.getAbsolutePath() + "'. It is not a file");
            return null;
        }
        byte[] imageBytes = Utils.readAllBytesOrExit(picture.toPath());
        Tensor<Float> image = constructAndExecuteGraphToNormalizeImage(imageBytes);
        float[] probabilities = executeGraph(execGraph, image);
        int bestLabelIdx = Utils.maxIndex(probabilities);
        return new TensorFlowResult(labels.get(bestLabelIdx), probabilities[bestLabelIdx], picture.getAbsolutePath());
    }
    
    public List<TensorFlowResult> evaluateDir(File imageDir) {
        List<TensorFlowResult> result = new ArrayList<TensorFlowResult>();
        File[] images = imageDir.listFiles();
        if (images == null || images.length==0) {
            System.err.println("No images found in selected imagedir '" + imageDir.getAbsolutePath() + "'.Nothing to do");
            return result;
        }
        for (File image: images) {
            result.add(evaluate(image));
        }
        return result;
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
                                  // Note: the 3 is the number of channels. It doesn't work, if we set it to 1 with the tested model
                              b.cast(b.decodeJpeg(input, 3), Float.class),
                              b.constant("make_batch", 0)),
                          b.constant("size", new int[] {H, W})),
                      b.constant("mean", mean)),
                  b.constant("scale", scale));
          try (Session s = new Session(g)) {
            //System.out.println("OP: " +   output.op().name());
            return s.runner()
                    .fetch(output.op().name())
                    .run()
                    .get(0)
                    .expect(Float.class);
          }
        }
    }
    
    /**
     * @param g an pretrained Tensor-Flow Graph
     * @param image an image represented as a Tensor<Float> to be used as input
     * @return the result of the evaluation of the image on the Graph
     */
    private static float[] executeGraph(Graph g, Tensor<Float> image) {
        try (Session s = new Session(g);
                Tensor<Float> result =
                        s.runner().feed("input", image).fetch(OutputOperationName).run().get(0).expect(Float.class)) {
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
