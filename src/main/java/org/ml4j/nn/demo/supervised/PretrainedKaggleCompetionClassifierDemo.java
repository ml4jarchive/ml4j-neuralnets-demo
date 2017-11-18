/*
 * Copyright 2017 the original author or authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
 * in compliance with the License. You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License
 * is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
 * or implied. See the License for the specific language governing permissions and limitations under
 * the License.
 */

package org.ml4j.nn.demo.supervised;

import org.ml4j.Matrix;
import org.ml4j.MatrixFactory;
import org.ml4j.imaging.targets.ImageDisplay;
import org.ml4j.jblas.JBlasMatrixFactory;
import org.ml4j.nn.FeedForwardNeuralNetworkContext;
import org.ml4j.nn.ForwardPropagation;
import org.ml4j.nn.activationfunctions.SigmoidActivationFunction;
import org.ml4j.nn.activationfunctions.SoftmaxActivationFunction;
import org.ml4j.nn.demo.base.supervised.SupervisedNeuralNetworkDemoBase;
import org.ml4j.nn.demo.util.KaggleMnistUtils;
import org.ml4j.nn.demo.util.KagglePixelFeaturesMatrixCsvDataExtractor;
import org.ml4j.nn.demo.util.SingleDigitLabelsMatrixCsvDataExtractor;
import org.ml4j.nn.layers.ConvolutionalFeedForwardLayerImpl;
import org.ml4j.nn.layers.DirectedLayerContext;
import org.ml4j.nn.layers.FeedForwardLayer;
import org.ml4j.nn.layers.FullyConnectedFeedForwardLayerImpl;
import org.ml4j.nn.layers.MaxPoolingFeedForwardLayerImpl;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.Neurons3D;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.neurons.NeuronsActivationFeatureOrientation;
import org.ml4j.nn.supervised.FeedForwardNeuralNetworkContextImpl;
import org.ml4j.nn.supervised.SupervisedFeedForwardNeuralNetwork;
import org.ml4j.nn.supervised.SupervisedFeedForwardNeuralNetworkImpl;
import org.ml4j.util.DoubleArrayMatrixLoader;
import org.ml4j.util.SerializationHelper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * A simple test harness to showcase a Classifier using pre-learned weights
 * from our Kaggle competition entry.
 * 
 * @author Michael Lavelle
 */
public class PretrainedKaggleCompetionClassifierDemo
    extends SupervisedNeuralNetworkDemoBase<SupervisedFeedForwardNeuralNetwork, 
    FeedForwardNeuralNetworkContext> {

  private static final Logger LOGGER = 
      LoggerFactory.getLogger(PretrainedKaggleCompetionClassifierDemo.class);

  public static void main(String[] args) throws Exception {
    PretrainedKaggleCompetionClassifierDemo demo = new PretrainedKaggleCompetionClassifierDemo();
    demo.runDemo();
  }

  @Override
  protected SupervisedFeedForwardNeuralNetwork 
      createSupervisedNeuralNetwork(int featureCount,
      boolean isBiasUnitIncluded) {

    // Construct a 5 layer Neural Network
    
    MatrixFactory matrixFactory = createMatrixFactory();
    
    // Load some pre-trained weights learned from our Kaggle competition entry.
    SerializationHelper helper = new SerializationHelper(
        PretrainedKaggleCompetionClassifierDemo.class.getClassLoader(), "pretrainedkaggleweights");

    Matrix layer1Weights =
        matrixFactory.createMatrix(helper.deserialize(double[][].class, "layer1"));
    Matrix layer3Weights =
        matrixFactory.createMatrix(helper.deserialize(double[][].class, "layer3"));
    Matrix layer4Weights =
        matrixFactory.createMatrix(helper.deserialize(double[][].class, "layer4"));
    Matrix layer5Weights =
        matrixFactory.createMatrix(helper.deserialize(double[][].class, "layer5"));

    
    // Construct a Neural Network in the same shape as our Kaggle entry.
    // Initialise each trainable layer with our pre-trained weights.
    
    FeedForwardLayer<?, ?> firstLayer = new ConvolutionalFeedForwardLayerImpl(
        new Neurons3D(28, 28 ,1, true), new Neurons3D(20, 20, 6, false), 
        new SigmoidActivationFunction(), matrixFactory, layer1Weights, false);
            
    // The max pooling layer that this NN was trained with originally was a legacy
    // implementation which scaled up the output activations by a factor of
    // 4.   Here we set the "scaleOutputs" property to true to account for this
    // legacy situation.  Normally we would set this property to false
    FeedForwardLayer<?, ?> secondLayer = 
        new MaxPoolingFeedForwardLayerImpl(new Neurons3D(20, 20, 6, false), 
            new Neurons3D(10, 10, 6, false), matrixFactory, true, false);
   
    FeedForwardLayer<?, ?> thirdLayer = 
        new FullyConnectedFeedForwardLayerImpl(new Neurons3D(10, 10, 6, true), 
            new Neurons3D(5, 5, 16, false), new SigmoidActivationFunction(), 
            matrixFactory, layer3Weights, false);
    
    FeedForwardLayer<?, ?> forthLayer = 
        new FullyConnectedFeedForwardLayerImpl(new Neurons(400, true), 
        new Neurons(100, false), new SigmoidActivationFunction(), matrixFactory,
        layer4Weights, false);
    
    FeedForwardLayer<?, ?> fifthLayer = 
        new FullyConnectedFeedForwardLayerImpl(new Neurons(100, true), 
        new Neurons(10, false), new SoftmaxActivationFunction(), matrixFactory,
        layer5Weights, false);

    return new SupervisedFeedForwardNeuralNetworkImpl(firstLayer, secondLayer,
        thirdLayer, forthLayer, fifthLayer);
  }

  @Override
  protected NeuronsActivation createTrainingDataNeuronActivations(MatrixFactory matrixFactory) {
    LOGGER.trace("Creating training data NeuronsActivation");
    
    DoubleArrayMatrixLoader loader = new DoubleArrayMatrixLoader(
            PretrainedKaggleCompetionClassifierDemo.class.getClassLoader());
    // Load Mnist data into double[][] matrices
    double[][] trainingDataMatrix = loader.loadDoubleMatrixFromCsv("train.csv",
            new KagglePixelFeaturesMatrixCsvDataExtractor(), 1, 1001);
    
    return new NeuronsActivation(matrixFactory.createMatrix(trainingDataMatrix), false,
        NeuronsActivationFeatureOrientation.COLUMNS_SPAN_FEATURE_SET);
  }

  @Override
  protected NeuronsActivation createTestSetDataNeuronActivations(MatrixFactory matrixFactory) {
    LOGGER.trace("Creating test data NeuronsActivation");
    
    DoubleArrayMatrixLoader loader = new DoubleArrayMatrixLoader(
        PretrainedKaggleCompetionClassifierDemo.class.getClassLoader());
    // Load Mnist data into double[][] matrices
    double[][] testDataMatrix = loader.loadDoubleMatrixFromCsv("train.csv",
        new KagglePixelFeaturesMatrixCsvDataExtractor(), 1001, 2001);

    return new NeuronsActivation(matrixFactory.createMatrix(testDataMatrix), false,
        NeuronsActivationFeatureOrientation.COLUMNS_SPAN_FEATURE_SET);
  }

  @Override
  protected MatrixFactory createMatrixFactory() {
    LOGGER.trace("Creating MatrixFactory");
    return new JBlasMatrixFactory();
  }

  @Override
  protected FeedForwardNeuralNetworkContext createTrainingContext(
      SupervisedFeedForwardNeuralNetwork supervisedNeuralNetwork, MatrixFactory matrixFactory) {
    LOGGER.trace("Creating FeedForwardNeuralNetworkContext");
    // Train from layer index 0 to the end layer
    FeedForwardNeuralNetworkContext context =
        new FeedForwardNeuralNetworkContextImpl(matrixFactory, 0, null);
    context.setTrainingIterations(0);
    return context;
  }
  
  @Override
  protected void showcaseTrainedNeuralNetworkOnTrainingSet(
      SupervisedFeedForwardNeuralNetwork neuralNetwork,
      NeuronsActivation testDataInputActivations, NeuronsActivation testDataLabelActivations, 
      MatrixFactory matrixFactory) throws Exception {
    
    // Create a context for the entire network
    FeedForwardNeuralNetworkContext accuracyContext =  
        new FeedForwardNeuralNetworkContextImpl(matrixFactory, 0, null);
   
    double classificationAccuracy = 
        neuralNetwork.getClassificationAccuracy(testDataInputActivations, 
            testDataLabelActivations, accuracyContext);
    
    LOGGER.info("Classification training set accuracy:" + classificationAccuracy);
    
  }

  @Override
  protected void showcaseTrainedNeuralNetworkOnTestSet(
      SupervisedFeedForwardNeuralNetwork neuralNetwork,
      NeuronsActivation testDataInputActivations, NeuronsActivation testDataLabelActivations, 
      MatrixFactory matrixFactory) throws Exception {

    // Create a context for the entire network
    FeedForwardNeuralNetworkContext accuracyContext =  
        new FeedForwardNeuralNetworkContextImpl(matrixFactory, 0, null);
   
    double classificationAccuracy = 
        neuralNetwork.getClassificationAccuracy(testDataInputActivations, 
            testDataLabelActivations, accuracyContext);
    
    LOGGER.info("Classification test set accuracy:" + classificationAccuracy);

    LOGGER.info("Showcasing trained Classifier...");

    // Create display for our demo
    ImageDisplay<Long> display = new ImageDisplay<Long>(280, 280);
    
    // Create a context for the first layer only
    FeedForwardNeuralNetworkContext autoEncoderNeuronVisualisationContext =  
        new FeedForwardNeuralNetworkContextImpl(matrixFactory, 0, 0);
    
    DirectedLayerContext hiddenNeuronInspectionContext = 
        autoEncoderNeuronVisualisationContext.getLayerContext(0);
    
    LOGGER.info("Drawing visualisations of patterns sought by the hidden neurons...");
    for (int j = 0; j < neuralNetwork.getFirstLayer().getOutputNeuronCount(); j++) {
      NeuronsActivation neuronActivationMaximisingActivation = neuralNetwork.getFirstLayer()
          .getOptimalInputForOutputNeuron(j, hiddenNeuronInspectionContext);
      double[] neuronActivationMaximisingFeatures =
          neuronActivationMaximisingActivation.getActivations().toArray();

      double[] intensities = new double[neuronActivationMaximisingFeatures.length];
      for (int i = 0; i < intensities.length; i++) {
        double val = neuronActivationMaximisingFeatures[i];
        double boundary = 0.02;
        intensities[i] = val < -boundary ? 0 : val > boundary ? 1 : 0.5;
      }
      KaggleMnistUtils.draw(intensities, display);
      Thread.sleep(5);
    }
    
    // Visualise the reconstructions of the input data
  
    LOGGER.info("Visualising data and classifying");
    for (int i = 0; i < 100; i++) {

      // For each element in our test set, obtain the compressed encoded features
      Matrix activations = testDataInputActivations.getActivations().getRow(i);
      
      NeuronsActivation orignalActivation = new NeuronsActivation(activations, false,
          NeuronsActivationFeatureOrientation.COLUMNS_SPAN_FEATURE_SET);

      KaggleMnistUtils.draw(orignalActivation.getActivations().toArray(), display);

      // Encode only through all Layers
      FeedForwardNeuralNetworkContext classifyingContext = 
          new FeedForwardNeuralNetworkContextImpl(matrixFactory, 0, null);

      ForwardPropagation forwardPropagtion = 
          neuralNetwork.forwardPropagate(orignalActivation, classifyingContext);

      double[] values = forwardPropagtion.getOutputs().getActivations().toArray();
         
      LOGGER.info("Classified digit as:" + getArgMaxIndex(values));
         
      Thread.sleep(2000);
    }
  }
  
  private int getArgMaxIndex(double[] data) {
    int maxValueIndex = 0;
    double maxValue = 0;
    for (int i = 0; i < data.length; i++) {
      if (data[i] > maxValue) {
        maxValue = data[i];
        maxValueIndex = i;
      }
    }
    return maxValueIndex;
  }

  @Override
  protected NeuronsActivation createTrainingLabelNeuronActivations(MatrixFactory matrixFactory) {
    DoubleArrayMatrixLoader loader = new DoubleArrayMatrixLoader(
        PretrainedKaggleCompetionClassifierDemo.class.getClassLoader());
    // Load Mnist data into double[][] matrices
    double[][] testDataMatrix = loader.loadDoubleMatrixFromCsv("train.csv",
        new SingleDigitLabelsMatrixCsvDataExtractor(), 1, 1001);
   
    return new NeuronsActivation(matrixFactory.createMatrix(testDataMatrix), false,
        NeuronsActivationFeatureOrientation.COLUMNS_SPAN_FEATURE_SET);
  }

  @Override
  protected NeuronsActivation createTestSetLabelNeuronActivations(MatrixFactory matrixFactory) {
    DoubleArrayMatrixLoader loader = new DoubleArrayMatrixLoader(
        PretrainedKaggleCompetionClassifierDemo.class.getClassLoader());
    // Load Mnist data into double[][] matrices
    double[][] testDataMatrix = loader.loadDoubleMatrixFromCsv("train.csv",
        new SingleDigitLabelsMatrixCsvDataExtractor(), 1001, 2001);
   
    return new NeuronsActivation(matrixFactory.createMatrix(testDataMatrix), false,
        NeuronsActivationFeatureOrientation.COLUMNS_SPAN_FEATURE_SET);
  }
}
