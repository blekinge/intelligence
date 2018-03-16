package dk.kb.deeplearning4j;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.io.labels.PathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.FlipImageTransform;
import org.datavec.image.transform.ImageTransform;
import org.datavec.image.transform.MultiImageTransform;
import org.datavec.image.transform.ShowImageTransform;
import org.datavec.image.transform.WarpImageTransform;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.MultipleEpochsIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.FileStatsStorage;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;

public class TrainFrontBackNetwork {

    public static void main(String[] args) throws IOException {
        if (args.length < 2) {
            System.err.println("Missing arguments: <dataset-dir> <trainpercentage>. Exiting program");
            System.exit(1);
        }
        String parentDirPath = args[0];
        File parentDir = new File(parentDirPath);
        if (!parentDir.isDirectory()) {
            System.err.println("Given dataset directory '" + parentDir.getAbsolutePath() + "' does not exist");
            System.err.println("Exiting program");
            System.exit(1);
        }
        int trainpercentage = Integer.parseInt(args[1]);
        if (trainpercentage >= 100 || trainpercentage <= 0) {
            System.err.println("Trainpercentage must be > 0 and < 100. Currently set to " + trainpercentage);
            System.err.println("Exiting program");
            System.exit(1);
        }
        System.out.println("Assigning " + trainpercentage + "% of data as training data");
        
        String[] allowedExtensions = new String[]{"jpg"};
        Random randNumGen = new Random();
        FileSplit filesInDir = new FileSplit(parentDir, allowedExtensions, randNumGen);
        
        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
        //Specifying a path filter to gives you fine tune control of the min/max cases to load for each class. Below is a bare bones version. Refer to javadocs for details
        BalancedPathFilter pathFilter = new BalancedPathFilter(randNumGen, allowedExtensions, labelMaker);
        
        InputSplit[] filesInDirSplit = filesInDir.sample(pathFilter, trainpercentage, 100-trainpercentage);
        InputSplit trainData = filesInDirSplit[0];
        System.out.println("Trainingdata is of size " + trainData.length());
        InputSplit testData = filesInDirSplit[1]; // This is currently not used
        System.out.println("Testdata is of size " + testData.length());
        int channels = 1;
        int heigth = 600;
        int width = 400;
        ImageRecordReader recordReader = new ImageRecordReader(heigth,width,channels,labelMaker);
        ImageTransform transform = new MultiImageTransform(randNumGen,new ShowImageTransform("Display - before "));
        //Initialize the record reader with the train data and the transform chain
        //recordReader.initialize(trainData,transform);
        recordReader.initialize(trainData); // without any transform
        
        int outputNum = recordReader.numLabels();
        System.out.println("numLabels: " + outputNum);
        
      //convert the record reader to an iterator for training - Refer to other examples for how to use an iterator
        int batchSize = 10; // Minibatch size. Here: The number of images to fetch for each call to dataIter.next().
        int labelIndex = 1; // Index of the label Writable (usually an IntWritable), as obtained by recordReader.next()
        // List<Writable> lw = recordReader.next();
        // then lw[0] =  NDArray shaped [1,3,50,50] (1, heightm width, channels)
        //      lw[0] =  label as integer.
/*
        DataSetIterator dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, labelIndex, outputNum);
        int iterations = 0;
        while (dataIter.hasNext()) {
            DataSet ds = dataIter.next();
            iterations++;
            //if (iterations % 100 == 0) {
            //    System.out.println("iteration " + iterations + ": " + ds);
            //}
            System.out.println(iterations);
            //System.out.println(ds);
            try {
                Thread.sleep(3000);                 //1000 milliseconds is one second.
            } catch(InterruptedException ex) {
                Thread.currentThread().interrupt();
            }
        }
        System.out.println("Completed " +  iterations + " iterations on training set");
        */
        try {
            doModelling(heigth, width, channels, labelMaker, trainData, testData, outputNum, parentDir);
        } catch (Throwable e) {
            System.err.println("Program stopped with exception: " + e.fillInStackTrace());
            System.exit(1);
        }
        System.out.println("Normal exit from program");
        System.exit(0);
    }
    
    
    public static void doModelling(int height, int width, int channels, PathLabelGenerator labelMaker,  InputSplit trainData, InputSplit testData, int numLabels, File parentDir) throws IOException {
        int iterations = 1;
        int epochs = 2;
        boolean save = true;
        int seed = 42;
        int batchSize = 20;
        boolean enableUIServer = false;
        ImageRecordReader recordReader = new ImageRecordReader(height, width, channels, labelMaker);
        //int numLabels = recordReader.numLabels();
        System.out.println("#labels: " + numLabels);
        DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
        //MultiLayerNetwork network = Models.getAlexNet(numLabels, seed, iterations).init();
        //MultiLayerNetwork network = Models.alexnetModel(seed, iterations, channels, height, width, numLabels);
        //MultiLayerNetwork network = Models.customModel1();
        //MultiLayerNetwork network = Models.getSimpleCnn(numLabels, seed, iterations);
        MultiLayerNetwork network = Models.lenetModel(seed, iterations, channels, numLabels, height, width);
        network.init();
        network.setListeners(new ScoreIterationListener());
        
        File f = new File(parentDir.getParentFile(), parentDir.getName() + ".db");
        StatsStorage statsStorage = new FileStatsStorage(f);

        if (enableUIServer) {
            UIServer uiServer = UIServer.getInstance();
            //StatsStorage statsStorage = new InMemoryStatsStorage();
            uiServer.attach(statsStorage);
        }
        network.setListeners(
                (IterationListener)
                new StatsListener( statsStorage),
                new ScoreIterationListener(iterations));
        DataSetIterator dataIter;
        MultipleEpochsIterator trainIter;

        System.out.println("Train model....without transformations");
        // Train without transformations
        recordReader.initialize(trainData, null);
        dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, numLabels);
        scaler.fit(dataIter);
        dataIter.setPreProcessor(scaler);
        trainIter = new MultipleEpochsIterator(epochs, dataIter);
        network.fit(trainIter);

        Random rng = new Random(seed);
        System.out.println("Train model....with transformations");    
        ImageTransform flipTransform1 = new FlipImageTransform(rng);
        ImageTransform flipTransform2 = new FlipImageTransform(new Random(123));
        ImageTransform warpTransform = new WarpImageTransform(rng, 42);
        //         ImageTransform colorTransform = new ColorConversionTransform(new Random(seed), COLOR_BGR2YCrCb);
        List<ImageTransform> transforms = Arrays.asList(new ImageTransform[]{flipTransform1, warpTransform, flipTransform2});
        for (ImageTransform transform : transforms) {
            System.out.print("\nTraining on transformation: " + transform.getClass().toString() + "\n\n");
            recordReader.initialize(trainData, transform);
            dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, numLabels);
            scaler.fit(dataIter);
            dataIter.setPreProcessor(scaler);
            trainIter = new MultipleEpochsIterator(epochs, dataIter);
            network.fit(trainIter);
        }
        System.out.print("Evaluating model....");
        recordReader.initialize(testData);
        dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, numLabels);
        scaler.fit(dataIter);
        dataIter.setPreProcessor(scaler);
        Evaluation eval = network.evaluate(dataIter);
        System.out.println("done.");
        System.out.println(eval.stats(true));

        // Example on how to get predict results with trained model. Result for first example in minibatch is printed
        dataIter.reset();
        DataSet testDataSet = dataIter.next();
        List<String> allClassLabels = recordReader.getLabels();
        int labelIndex = testDataSet.getLabels().argMax(1).getInt(0);
        int[] predictedClasses = network.predict(testDataSet.getFeatures());
        String expectedResult = allClassLabels.get(labelIndex);
        String modelPrediction = allClassLabels.get(predictedClasses[0]);
        System.out.print("\nFor a single example that is labeled " + expectedResult + " the model predicted " + modelPrediction + "\n\n");

        if (save) {
            Models.saveModel(parentDir, network);
        }
        System.out.print("****************Example finished********************");

    }
}
