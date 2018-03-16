package dk.kb.deeplearning4j;

import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;

import org.apache.commons.io.FileUtils;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;


/**
 * Using the following issue as an example to test model
 * https://github.com/deeplearning4j/dl4j-examples/issues/351
 */

public class UseModel {

    public static void main(String[] args) throws IOException {
        if (args.length < 3) {
            System.out.println("Missing arguments: <model> <label-file> <image-dir>|<image-file>");
            System.exit(1);
        }
        String path = args[0];
        String labelfilePath = args[1];
        String filepath = args[2];
        File labelFile = new File(labelfilePath);
        File model = new File(path);
        File imageDir = new File(filepath);
        if (!model.isFile()) {
            System.out.println("The given model '" + model.getAbsolutePath() + "' does not exist");;
            System.exit(1);
        }
        if (!imageDir.exists()) {
            System.out.println("The given image-directory or imagefile '" + imageDir.getAbsolutePath() + "' does not exist");;
            System.exit(1);
        }
        boolean singleFile = imageDir.isFile();
        MultiLayerNetwork nn = Models.readModel(model);
        String[] labels = readLabelFile(labelFile);
        for (int i=0; i< labels.length; i++) {
            System.out.println("Assuming labelIndex["+ i + "] refers to '" + labels[i] + "'");
        }
        if (singleFile) {
            runModelOnImageFile(nn, imageDir, labels);
        } else {
            runModelOnImagedir(nn, imageDir, labels);
        }
        
     }
    
    private static void runModelOnImageFile(MultiLayerNetwork nn, File imageFile, String[] labels) throws IOException {
        NativeImageLoader loader = new NativeImageLoader(600, 400, 1);
        INDArray output = evaluateImage(nn, loader, imageFile);
        Result r = Result.getResult(output, imageFile.getAbsolutePath());
        String s = "File '" +  r.getObjectName() + "' matched category '" + labels[r.getLabelIndex()] + "' with accuracy " + r.getAccuracy();
        System.out.println(s);
    }

    private static String[] readLabelFile(File labelFile) throws IOException {
        List<String> labelsAsLines = FileUtils.readLines(labelFile);
        return labelsAsLines.toArray(new String[labelsAsLines.size()]);
    }

    public static void runModelOnImagedir(MultiLayerNetwork network, File imageDir, String[] labels) throws IOException {
        NativeImageLoader loader = new NativeImageLoader(600, 400, 1);
        File[] images = imageDir.listFiles();
        if (images == null || images.length==0) {
            System.err.println("No images found in selected imagedir '" + imageDir.getAbsolutePath() + "'.Nothing to do");
            System.exit(0);
        } 
        System.out.println("Found " + images.length + " to evaluate");
        long started = System.currentTimeMillis();
        Map<Integer, HashSet<Result>> labelMap = new HashMap<Integer, HashSet<Result>>();
        for (int j=0; j < labels.length; j++) {
            labelMap.put(j, new HashSet<Result>());
        }
        for(int i=0; i < images.length; i++) {
            INDArray output = evaluateImage(network, loader, images[i]);
            Result r = Result.getResult(output, images[i].getAbsolutePath());
            HashSet<Result> rset = labelMap.get(r.getLabelIndex());
            rset.add(r);
            labelMap.put(r.getLabelIndex(), rset);
        }
        long timeUsedInSeconds = (System.currentTimeMillis() - started)/1000;
        
        File outputDir = new File(imageDir.getParentFile(), "output-" + System.currentTimeMillis());
        outputDir.mkdir();
        System.out.println("Tested " + images.length + " images. Evaluation time (secs): " +  timeUsedInSeconds); 
        System.out.println("Saving result to " + outputDir.getAbsolutePath());
        for (int j=0; j < labels.length; j++) {
            System.out.println(" category '" + labels[j] + "': " + labelMap.get(j).size());
            File catFile = new File(outputDir, labels[j] + ".txt");
            writeToFile(labels[j], labelMap.get(j), catFile);
        }
    }
    
    private static void writeToFile(String category, HashSet<Result> hashSet,
            File catFile) throws IOException {
        Charset charset = Charset.forName("UTF-8");
        String s = null;
        BufferedWriter writer = null;
        try {
            writer = Files.newBufferedWriter(catFile.toPath(), charset);
            s = "#entries matching category '" + category + "': " + hashSet.size();
            writer.write(s, 0, s.length());
            writer.newLine();
            for (Result r: hashSet) {
                s= "File '" +  r.getObjectName() + "' matched this category with accuracy " + r.getAccuracy();
                writer.write(s, 0, s.length());
                writer.newLine();
            }
        } catch (IOException x) {
            System.err.format("IOException: %s%n", x);
        } finally {
            writer.flush();
            writer.close();
        }   
    }

    public static INDArray evaluateImage(MultiLayerNetwork network, NativeImageLoader loader, File imageFile) throws IOException {
        INDArray image = loader.asMatrix(imageFile);
        // 0-255
        // 0-1
        DataNormalization scaler = new ImagePreProcessingScaler(0,1);
        scaler.transform(image);
        return network.output(image);
    }

}
