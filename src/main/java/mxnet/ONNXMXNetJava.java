package mxnet;

import org.apache.mxnet.infer.javaapi.Predictor;
import org.apache.mxnet.javaapi.*;
import org.apache.mxnet.ResourceScope;
import org.kohsuke.args4j.CmdLineParser;
import org.kohsuke.args4j.Option;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Collections;

public class ONNXMXNetJava {
    @Option(name = "--model-path-prefix", usage = "input model directory and prefix of the model")
    private String modelPathPrefix = System.getProperty("user.dir") + "/model/vgg16";
    @Option(name = "--input-image", usage = "the input image")
    private String inputImagePath = System.getProperty("user.dir") + "/data/Penguin.jpg";

    final static Logger logger = LoggerFactory.getLogger(ONNXMXNetJava.class);

    public static void main(String[] args) {
        ONNXMXNetJava inst = new ONNXMXNetJava();
        CmdLineParser parser  = new CmdLineParser(inst);
        try {
            parser.parseArgument(args);
        } catch (Exception e) {
            logger.error(e.getMessage(), e);
            parser.printUsage(System.err);
            System.exit(1);
        }

        List<Context> inferenceContext = Collections.singletonList(Context.cpu());

        Shape inputShape = new Shape(new int[] {1, 3, 224, 224});

        List<DataDesc> inputDescriptors = new ArrayList<>();
        inputDescriptors.add(new DataDesc("Input_0", inputShape, DType.Float32(), "NCHW"));

        Predictor predictor = new Predictor(inst.modelPathPrefix, inputDescriptors, inferenceContext, 0);

        try(ResourceScope scope = new ResourceScope()) {
            NDArray img = Image.imRead(inst.inputImagePath, 1, true);
            img = Image.imResize(img, 224, 224);
            NDArray nd = img;
            nd = NDArray.transpose(nd, new Shape(new int[]{2, 0, 1}), null)[0];
            nd = NDArray.expand_dims(nd, 0, null)[0];
            nd = nd.asType(DType.Float32());
            List<NDArray> ndList = Collections.singletonList(nd);
            List<NDArray> ndResult = predictor.predictWithNDArray(ndList);
            try {
                System.out.println("Prediction for " + inst.inputImagePath);
                System.out.println(printMaximumClass(ndResult.get(0).toArray(), inst.modelPathPrefix));
            } catch (IOException e) {
                System.err.println(e);
            }
        }
    }


    private static String printMaximumClass(float[] probabilities,
                                            String modelPathPrefix) throws IOException {
        String synsetFilePath = modelPathPrefix.substring(0,
                1 + modelPathPrefix.lastIndexOf(File.separator)) + "/synset.txt";
        BufferedReader reader = new BufferedReader(new FileReader(synsetFilePath));
        ArrayList<String> list = new ArrayList<>();
        String line = reader.readLine();

        while (line != null){
            list.add(line);
            line = reader.readLine();
        }
        reader.close();

        int maxIdx = 0;
        for (int i = 1;i<probabilities.length;i++) {
            if (probabilities[i] > probabilities[maxIdx]) {
                maxIdx = i;
            }
        }

        return "Probability : " + probabilities[maxIdx] + " Class : " + list.get(maxIdx) ;
    }
}
