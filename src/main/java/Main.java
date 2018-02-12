import org.datavec.api.records.listener.impl.LogRecordListener;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;
import java.util.Random;

public class Main {

    private static final String NEURAL_NETWORK_FILE_NAME = "MyComputationGraph.zip";

    private static int clipLengthSecs = 120;


    public static void main(String[] args) throws IOException {
        int rngseed = 123;
        Random randNumGen = new Random(rngseed);

        System.out.println("Instantiating important stuff...");
        FftUtil fftUtil = new FftUtil(clipLengthSecs, 8192);

        System.out.println("Running FFTs");
        File trainingDirectory = new File("/Users/danieljackson/Desktop/nnresources/TrainingData");
        File testDirectory = new File("/Users/danieljackson/Desktop/nnresources/TestData");

        convertImagesAt(trainingDirectory, fftUtil);
        convertImagesAt(testDirectory, fftUtil);
        FileSplit train = new FileSplit(trainingDirectory, NativeImageLoader.ALLOWED_FORMATS, randNumGen);
        FileSplit test = new FileSplit(testDirectory, NativeImageLoader.ALLOWED_FORMATS, randNumGen);

        FileNameLabelGenerator labelMaker = new FileNameLabelGenerator();
        ImageRecordReader recordReader = new ImageRecordReader(fftUtil.getMatrixHeight(), fftUtil.getMatrixWidth(), 3, labelMaker);

        recordReader.initialize(train);
        recordReader.setListeners(new LogRecordListener());

        DataSetIterator dataIter = new RecordReaderDataSetIterator(recordReader, 8, 1, 2);

        DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
        scaler.fit(dataIter);
        dataIter.setPreProcessor(scaler);


        File netDirectory = new File("/Users/danieljackson/Desktop/nnresources/network");
        MultiLayerNetwork model = loadMultiLayerNetwork(netDirectory, fftUtil, dataIter);

        System.out.println("Starting testing...");

        recordReader.reset();

        recordReader.initialize(test);
        DataSetIterator testIter = new RecordReaderDataSetIterator(recordReader,1,1,2);
        scaler.fit(testIter);
        testIter.setPreProcessor(scaler);

        // Create Eval object with 2 possible classes
        Evaluation eval = new Evaluation(2);

        while(testIter.hasNext()){
            DataSet next = testIter.next();
            INDArray output = model.output(next.getFeatures());
            eval.eval(next.getLabels(),output);

        }

        System.out.println(eval.stats());
        System.out.println(eval.getLabelsList());
    }

    private static MultiLayerNetwork loadMultiLayerNetwork(File file, FftUtil fftUtil, DataSetIterator dataSetIterator) throws IOException {
        for (File networkFile : file.listFiles()) {

            if (networkFile.getName().equalsIgnoreCase(NEURAL_NETWORK_FILE_NAME)) {
                System.out.println("Network file found, skipping training");
                return ModelSerializer.restoreMultiLayerNetwork(networkFile);
            }
        }

        ConvolutionLayer layer0 = new ConvolutionLayer.Builder(5, 5)
                .nIn(3)
                .nOut(16)
                .stride(1, 1)
                .padding(2, 2)
                .weightInit(WeightInit.XAVIER)
                .name("First convolution layer")
                .activation(Activation.RELU)
                .build();

        SubsamplingLayer layer1 = new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                .kernelSize(2, 2)
                .stride(2, 2)
                .name("First subsampling layer")
                .build();

        ConvolutionLayer layer2 = new ConvolutionLayer.Builder(5, 5)
                .nOut(20)
                .stride(1, 1)
                .padding(2, 2)
                .weightInit(WeightInit.XAVIER)
                .name("Second convolution layer")
                .activation(Activation.RELU)
                .build();

        SubsamplingLayer layer3 = new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                .kernelSize(2, 2)
                .stride(2, 2)
                .name("Second subsampling layer")
                .build();

        ConvolutionLayer layer4 = new ConvolutionLayer.Builder(5, 5)
                .nOut(20)
                .stride(1, 1)
                .padding(2, 2)
                .weightInit(WeightInit.XAVIER)
                .name("Third convolution layer")
                .activation(Activation.RELU)
                .build();

        SubsamplingLayer layer5 = new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                .kernelSize(2, 2)
                .stride(2, 2)
                .name("Third subsampling layer")
                .build();

        OutputLayer layer6 = new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .activation(Activation.SOFTMAX)
                .weightInit(WeightInit.XAVIER)
                .name("Output")
                .nOut(2)
                .build();

        MultiLayerConfiguration configuration = new NeuralNetConfiguration.Builder()
                .seed(12345)
                .iterations(15)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .learningRate(0.001)
                .regularization(true)
                .l2(0.0004)
                .updater(Updater.NESTEROVS)
                .list()
                .layer(0, layer0)
                .layer(1, layer1)
                .layer(2, layer2)
                .layer(3, layer3)
                .layer(4, layer4)
                .layer(5, layer5)
                .layer(6, layer6)
                .pretrain(false)
                .backprop(true)
                .setInputType(InputType.convolutional(fftUtil.getMatrixWidth(), fftUtil.getMatrixHeight(), 1))
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(configuration);
        model.init();


        model.setListeners(new Listener());

        System.out.println("*****TRAIN MODEL********");
        for (int i = 0; i < 2; i++) {
            model.fit(dataSetIterator);
        }

        saveNetwork(model, file);

        return model;
    }

    private static void saveNetwork(MultiLayerNetwork network, File parentDirectory) throws IOException {
        System.out.println("Saving network to disc");
        File locationToSave = new File(parentDirectory.getAbsolutePath() + "/" + NEURAL_NETWORK_FILE_NAME);
        ModelSerializer.writeModel(network, locationToSave, true);
    }

    private static void convertImagesAt(File file, FftUtil fftUtil) {
        for (File file1 : file.listFiles()) {
            if (file1.listFiles() != null) {
                SpectrographLoader loader = new SpectrographLoader(file1, fftUtil, FftUtil.Strategy.MIDDLE);
                while (loader.hasNext()) {
                    loader.next();
                }
            }
        }
    }
}
