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

import java.util.Arrays;

import org.ml4j.Matrix;
import org.ml4j.MatrixFactory;
import org.ml4j.imaging.targets.ImageDisplay;
import org.ml4j.jblas.JBlasRowMajorMatrixFactory;
import org.ml4j.nn.ForwardPropagation;
import org.ml4j.nn.LayeredFeedForwardNeuralNetworkContext;
import org.ml4j.nn.activationfunctions.DefaultSigmoidActivationFunctionImpl;
import org.ml4j.nn.activationfunctions.DefaultSoftmaxActivationFunctionImpl;
import org.ml4j.nn.activationfunctions.factories.DifferentiableActivationFunctionFactory;
import org.ml4j.nn.axons.factories.AxonsFactory;
import org.ml4j.nn.components.factories.DirectedComponentFactory;
import org.ml4j.nn.demo.base.supervised.SupervisedNeuralNetworkDemoBase;
import org.ml4j.nn.demo.util.KaggleMnistUtils;
import org.ml4j.nn.demo.util.KagglePixelFeaturesMatrixCsvDataExtractor;
import org.ml4j.nn.demo.util.SingleDigitLabelsMatrixCsvDataExtractor;
import org.ml4j.nn.factories.DefaultAxonsFactoryImpl;
import org.ml4j.nn.factories.DefaultDifferentiableActivationFunctionFactory;
import org.ml4j.nn.factories.DefaultDirectedComponentFactoryImpl;
import org.ml4j.nn.layers.ConvolutionalFeedForwardLayerImpl;
import org.ml4j.nn.layers.DirectedLayerContext;
import org.ml4j.nn.layers.FeedForwardLayer;
import org.ml4j.nn.layers.FullyConnectedFeedForwardLayerImpl;
import org.ml4j.nn.layers.MaxPoolingFeedForwardLayerImpl;
import org.ml4j.nn.neurons.Neurons1D;
import org.ml4j.nn.neurons.Neurons3D;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.neurons.NeuronsActivationFeatureOrientation;
import org.ml4j.nn.neurons.NeuronsActivationImpl;
import org.ml4j.nn.supervised.LayeredFeedForwardNeuralNetworkContextImpl;
import org.ml4j.nn.supervised.LayeredSupervisedFeedForwardNeuralNetwork;
import org.ml4j.nn.supervised.LayeredSupervisedFeedForwardNeuralNetworkImpl;
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
    extends SupervisedNeuralNetworkDemoBase<LayeredSupervisedFeedForwardNeuralNetwork, 
    LayeredFeedForwardNeuralNetworkContext> {

  private static final Logger LOGGER = 
      LoggerFactory.getLogger(PretrainedKaggleCompetionClassifierDemo.class);

  public static void main(String[] args) throws Exception {
    PretrainedKaggleCompetionClassifierDemo demo = new PretrainedKaggleCompetionClassifierDemo();
    demo.runDemo();
  }
 
  private float[][] toFloatArray(double[][] data) {
	  float[][] result = new float[data.length][data[0].length];
	  for (int r = 0; r < data.length; r++) {
		  for (int c = 0; c < data[0].length; c++) {
			  result[r][c] = (float)data[r][c];
		  }
	  }
	  return result;
  }	

  @Override
  protected LayeredSupervisedFeedForwardNeuralNetwork createSupervisedNeuralNetwork(int featureCount) {

    // Construct a 5 layer Neural Network
    
    MatrixFactory matrixFactory = createMatrixFactory();
    
    AxonsFactory axonsFactory = new DefaultAxonsFactoryImpl(matrixFactory);
    
    DifferentiableActivationFunctionFactory activationFunctionFactory = new DefaultDifferentiableActivationFunctionFactory();
    
    DirectedComponentFactory directedComponentFactory = new DefaultDirectedComponentFactoryImpl(matrixFactory, axonsFactory, activationFunctionFactory);
    
    DifferentiableActivationFunctionFactory differentiableActivationFunctionFactory = new DefaultDifferentiableActivationFunctionFactory();

    
    // Load some pre-trained weights learned from our Kaggle competition entry.
    SerializationHelper helper = new SerializationHelper(
        PretrainedKaggleCompetionClassifierDemo.class.getClassLoader(), "pretrainedweights");

    Matrix layer1Weights = matrixFactory.createMatrixFromRowsByRowsArray(6, 81, helper.deserialize(float[].class, "layer1Weights"));
    Matrix layer1Biases = matrixFactory.createMatrixFromRowsByRowsArray(6, 1,  helper.deserialize(float[].class, "layer1Biases"));

    Matrix layer3Weights = matrixFactory.createMatrixFromRowsByRowsArray(400, 600, helper.deserialize(float[].class, "layer3Weights"));
    Matrix layer3Biases = matrixFactory.createMatrixFromRowsByRowsArray(400, 1,  helper.deserialize(float[].class, "layer3Biases"));

    Matrix layer4Weights = matrixFactory.createMatrixFromRowsByRowsArray(100, 400, helper.deserialize(float[].class, "layer4Weights"));
    Matrix layer4Biases = matrixFactory.createMatrixFromRowsByRowsArray(100, 1,  helper.deserialize(float[].class, "layer4Biases"));
    
    Matrix layer5Weights = matrixFactory.createMatrixFromRowsByRowsArray(10, 100, helper.deserialize(float[].class, "layer5Weights"));
    Matrix layer5Biases = matrixFactory.createMatrixFromRowsByRowsArray(10, 1,  helper.deserialize(float[].class, "layer5Biases"));
    
    // Construct a Neural Network in the same shape as our Kaggle entry.
    // Initialise each trainable layer with our pre-trained weights.
    
    FeedForwardLayer<?, ?> firstLayer = new ConvolutionalFeedForwardLayerImpl(
        directedComponentFactory, axonsFactory, new Neurons3D(28, 28 ,1, true), new Neurons3D(20, 20, 6, false), 
        new DefaultSigmoidActivationFunctionImpl(), matrixFactory, layer1Weights, layer1Biases, false);
            
    // The max pooling layer that this NN was trained with originally was a legacy
    // implementation which scaled up the output activations by a factor of
    // 4.   Here we set the "scaleOutputs" property to true to account for this
    // legacy situation.  Normally we would set this property to false
    FeedForwardLayer<?, ?> secondLayer = 
        new MaxPoolingFeedForwardLayerImpl(directedComponentFactory, axonsFactory, differentiableActivationFunctionFactory, new Neurons3D(20, 20, 6, false), 
            new Neurons3D(10, 10, 6, false), matrixFactory, true, false, 2);
   
    FeedForwardLayer<?, ?> thirdLayer = 
        new FullyConnectedFeedForwardLayerImpl(directedComponentFactory, axonsFactory, new Neurons3D(10, 10, 6, true), 
            new Neurons3D(5, 5, 16, false), new DefaultSigmoidActivationFunctionImpl(), 
            matrixFactory, layer3Weights, layer3Biases, false);
    
    FeedForwardLayer<?, ?> forthLayer = 
        new FullyConnectedFeedForwardLayerImpl(directedComponentFactory, axonsFactory, new Neurons1D(400, true), 
        new Neurons1D(100, false), new DefaultSigmoidActivationFunctionImpl(), matrixFactory,
        layer4Weights, layer4Biases, false);
    
    FeedForwardLayer<?, ?> fifthLayer = 
        new FullyConnectedFeedForwardLayerImpl(directedComponentFactory, axonsFactory, new Neurons1D(100, true), 
        new Neurons1D(10, false), new DefaultSoftmaxActivationFunctionImpl(), matrixFactory,
        layer5Weights, layer5Biases, false);

    return new LayeredSupervisedFeedForwardNeuralNetworkImpl(directedComponentFactory, Arrays.asList(firstLayer, secondLayer,
        thirdLayer, forthLayer, fifthLayer));
  }

  @Override
  protected NeuronsActivation createTrainingDataNeuronActivations(MatrixFactory matrixFactory) {
    LOGGER.trace("Creating training data NeuronsActivation");
    
    DoubleArrayMatrixLoader loader = new DoubleArrayMatrixLoader(
            PretrainedKaggleCompetionClassifierDemo.class.getClassLoader());
    // Load Mnist data into double[][] matrices
    float[][] trainingDataMatrix = toFloatArray(loader.loadDoubleMatrixFromCsv("train.csv",
            new KagglePixelFeaturesMatrixCsvDataExtractor(), 1, 1001));
    
    NeuronsActivation act = new NeuronsActivationImpl(new Neurons1D(trainingDataMatrix[0].length, false), matrixFactory.createMatrixFromRows(trainingDataMatrix).transpose(),
        NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET);
    
    return act;
    
   // return act.createDownstreamActivation(act.getNewActivationsFormat(matrixFactory, new Neurons3D(28, 28 ,1, true)));
  }

  @Override
  protected NeuronsActivation createTestSetDataNeuronActivations(MatrixFactory matrixFactory) {
    LOGGER.trace("Creating test data NeuronsActivation");
    
    DoubleArrayMatrixLoader loader = new DoubleArrayMatrixLoader(
        PretrainedKaggleCompetionClassifierDemo.class.getClassLoader());
    // Load Mnist data into double[][] matrices
    float[][] testDataMatrix = toFloatArray(loader.loadDoubleMatrixFromCsv("train.csv",
        new KagglePixelFeaturesMatrixCsvDataExtractor(), 1001, 2001));

    return new NeuronsActivationImpl(new Neurons1D(testDataMatrix[0].length, false),  matrixFactory.createMatrixFromRows(testDataMatrix).transpose(),
        NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET);
  }

  @Override
  protected MatrixFactory createMatrixFactory() {
    LOGGER.trace("Creating MatrixFactory");
    return new JBlasRowMajorMatrixFactory();
  }

  
  
  @Override
  protected void showcaseTrainedNeuralNetworkOnTrainingSet(
      LayeredSupervisedFeedForwardNeuralNetwork neuralNetwork,
      NeuronsActivation testDataInputActivations, NeuronsActivation testDataLabelActivations, 
      MatrixFactory matrixFactory) throws Exception {
    
    // Create a context for the entire network
	  LayeredFeedForwardNeuralNetworkContext accuracyContext =  
        new LayeredFeedForwardNeuralNetworkContextImpl(matrixFactory, 0, null, false);
   
    double classificationAccuracy = 
        neuralNetwork.getClassificationAccuracy(testDataInputActivations, 
            testDataLabelActivations, accuracyContext);
    
    LOGGER.info("Classification training set accuracy:" + classificationAccuracy);
    
  }

  @Override
  protected void showcaseTrainedNeuralNetworkOnTestSet(
      LayeredSupervisedFeedForwardNeuralNetwork neuralNetwork,
      NeuronsActivation testDataInputActivations, NeuronsActivation testDataLabelActivations, 
      MatrixFactory matrixFactory) throws Exception {
	  
	  // Temporarily prevent the test set activations being closed throughout this demo. TODO. Implement auto-locking of first input
	  // in activation chain to prevent closure by default.
	  testDataInputActivations.setImmutable(true);

    // Create a context for the entire network
	  LayeredFeedForwardNeuralNetworkContext accuracyContext =  
        new LayeredFeedForwardNeuralNetworkContextImpl(matrixFactory, 0, null, false);
   
    double classificationAccuracy = 
        neuralNetwork.getClassificationAccuracy(testDataInputActivations, 
            testDataLabelActivations, accuracyContext);
    
    LOGGER.info("Classification test set accuracy:" + classificationAccuracy);

    LOGGER.info("Showcasing trained Classifier...");

    // Create display for our demo
    ImageDisplay<Long> display = new ImageDisplay<Long>(280, 280);
    
    // Create a context for the first layer only
    LayeredFeedForwardNeuralNetworkContext autoEncoderNeuronVisualisationContext =  
        new LayeredFeedForwardNeuralNetworkContextImpl(matrixFactory, 0, null, false);
    
    DirectedLayerContext hiddenNeuronInspectionContext = 
        autoEncoderNeuronVisualisationContext.getLayerContext(0);

    LOGGER.info("Drawing visualisations of patterns sought by the hidden neurons...");
    for (int j = 0; j < 6; j++) {
      NeuronsActivation neuronActivationMaximisingActivation = neuralNetwork.getFirstLayer()
          .getOptimalInputForOutputNeuron(j, hiddenNeuronInspectionContext);
      float[] neuronActivationMaximisingFeatures =
          neuronActivationMaximisingActivation.getActivations(matrixFactory).getRowByRowArray();

      float[] intensities = new float[neuronActivationMaximisingFeatures.length];
      for (int i = 0; i < intensities.length; i++) {
        float val = neuronActivationMaximisingFeatures[i];
        float boundary = 0.02f;
        intensities[i] = val < -boundary ? 0f : val > boundary ? 1f : 0.5f;
      }
      KaggleMnistUtils.drawNineByNine(intensities, display);
      Thread.sleep(1000);
    }
   
    
    // Visualise the reconstructions of the input data
  
    LOGGER.info("Visualising data and classifying");
    for (int i = 0; i < 100; i++) {

      // For each element in our test set, obtain the compressed encoded features
      Matrix activations = testDataInputActivations.getActivations(matrixFactory).getColumn(i);
      
      NeuronsActivation orignalActivation = new NeuronsActivationImpl(new Neurons1D(activations.getRows(), false), activations,
          NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET);

      KaggleMnistUtils.draw(orignalActivation.getActivations(matrixFactory).toColumnByColumnArray(), display);

      // Encode only through all Layers
      LayeredFeedForwardNeuralNetworkContext classifyingContext = 
          new LayeredFeedForwardNeuralNetworkContextImpl(matrixFactory, 0, null, false);

      ForwardPropagation forwardPropagtion = 
          neuralNetwork.forwardPropagate(orignalActivation, classifyingContext);

      float[] values = forwardPropagtion.getOutput().getActivations(matrixFactory).toColumnByColumnArray();
         
      LOGGER.info("Classified digit as:" + getArgMaxIndex(values));
         
      Thread.sleep(2000);
    }
  }
  
  private int getArgMaxIndex(float[] data) {
    int maxValueIndex = 0;
    float maxValue = 0;
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
    float[][] testDataMatrix = toFloatArray(loader.loadDoubleMatrixFromCsv("train.csv",
        new SingleDigitLabelsMatrixCsvDataExtractor(), 1, 1001));
   
    return new NeuronsActivationImpl(new Neurons1D(testDataMatrix[0].length, false), matrixFactory.createMatrixFromRows(testDataMatrix).transpose(),
        NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET);
  }

  @Override
  protected NeuronsActivation createTestSetLabelNeuronActivations(MatrixFactory matrixFactory) {
    DoubleArrayMatrixLoader loader = new DoubleArrayMatrixLoader(
        PretrainedKaggleCompetionClassifierDemo.class.getClassLoader());
    // Load Mnist data into double[][] matrices
    float[][] testDataMatrix = toFloatArray(loader.loadDoubleMatrixFromCsv("train.csv",
        new SingleDigitLabelsMatrixCsvDataExtractor(), 1001, 2001));
   
    return new NeuronsActivationImpl(new Neurons1D(testDataMatrix[0].length, false), matrixFactory.createMatrixFromRows(testDataMatrix).transpose(),
        NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET);
  }



@Override
protected LayeredFeedForwardNeuralNetworkContext createTrainingContext(
		LayeredSupervisedFeedForwardNeuralNetwork supervisedNeuralNetwork, MatrixFactory matrixFactory) {
	  LOGGER.trace("Creating FeedForwardNeuralNetworkContext");
	  // Train from layer index 0 to the end layer
	  LayeredFeedForwardNeuralNetworkContext context =
	      new LayeredFeedForwardNeuralNetworkContextImpl(matrixFactory, 0, null, true);
	  context.setTrainingEpochs(5);
	  context.setTrainingLearningRate(0.01f);
	  context.getLayerContext(0).getSynapsesContext(0).getAxonsContext(0, 0).withLeftHandInputDropoutKeepProbability(0.5f);
	  return context;
}
}
