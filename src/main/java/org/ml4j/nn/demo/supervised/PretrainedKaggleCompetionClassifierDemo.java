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
import org.ml4j.jblas.JBlasRowMajorMatrixFactoryOptimised;
import org.ml4j.nn.ForwardPropagation;
import org.ml4j.nn.LayeredFeedForwardNeuralNetworkContext;
import org.ml4j.nn.activationfunctions.ActivationFunctionBaseType;
import org.ml4j.nn.activationfunctions.factories.DifferentiableActivationFunctionFactory;
import org.ml4j.nn.axons.BiasMatrix;
import org.ml4j.nn.axons.BiasMatrixImpl;
import org.ml4j.nn.axons.WeightsFormatImpl;
import org.ml4j.nn.axons.WeightsMatrix;
import org.ml4j.nn.axons.WeightsMatrixImpl;
import org.ml4j.nn.axons.WeightsMatrixOrientation;
import org.ml4j.nn.axons.factories.AxonsFactory;
import org.ml4j.nn.components.factories.DirectedComponentFactory;
import org.ml4j.nn.demo.base.supervised.SupervisedNeuralNetworkDemoBase;
import org.ml4j.nn.demo.util.KaggleMnistUtils;
import org.ml4j.nn.demo.util.KagglePixelFeaturesMatrixCsvDataExtractor;
import org.ml4j.nn.demo.util.SingleDigitLabelsMatrixCsvDataExtractor;
import org.ml4j.nn.factories.DefaultAxonsFactoryImpl;
import org.ml4j.nn.factories.DefaultDifferentiableActivationFunctionFactory;
import org.ml4j.nn.factories.DefaultDirectedComponentFactoryImpl;
import org.ml4j.nn.layers.DefaultDirectedLayerFactory;
import org.ml4j.nn.layers.DirectedLayerContext;
import org.ml4j.nn.layers.DirectedLayerFactory;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.Neurons3D;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.neurons.NeuronsActivationImpl;
import org.ml4j.nn.neurons.format.ImageNeuronsActivationFormat;
import org.ml4j.nn.neurons.format.NeuronsActivationFormat;
import org.ml4j.nn.neurons.format.features.Dimension;
import org.ml4j.nn.sessions.factories.DefaultSessionFactory;
import org.ml4j.nn.sessions.factories.DefaultSessionFactoryImpl;
import org.ml4j.nn.supervised.DefaultLayeredSupervisedFeedForwardNeuralNetworkFactory;
import org.ml4j.nn.supervised.DefaultSupervisedFeedForwardNeuralNetworkFactory;
import org.ml4j.nn.supervised.LayeredFeedForwardNeuralNetworkContextImpl;
import org.ml4j.nn.supervised.LayeredSupervisedFeedForwardNeuralNetwork;
import org.ml4j.nn.supervised.LayeredSupervisedFeedForwardNeuralNetworkFactory;
import org.ml4j.nn.supervised.SupervisedFeedForwardNeuralNetworkFactory;
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

    DirectedLayerFactory layerFactory = new DefaultDirectedLayerFactory(axonsFactory, differentiableActivationFunctionFactory, directedComponentFactory);

    
    // Load some pre-trained weights learned from our Kaggle competition entry.
    SerializationHelper helper = new SerializationHelper(
        PretrainedKaggleCompetionClassifierDemo.class.getClassLoader(), "pretrainedweights");

		WeightsMatrix layer1Weights = new WeightsMatrixImpl(
				matrixFactory.createMatrixFromRowsByRowsArray(6, 81,
						helper.deserialize(float[].class, "layer1Weights")),
				new WeightsFormatImpl(
						Arrays.asList(Dimension.INPUT_DEPTH, Dimension.FILTER_HEIGHT, Dimension.FILTER_WIDTH),
						Arrays.asList(Dimension.OUTPUT_DEPTH), WeightsMatrixOrientation.ROWS_SPAN_OUTPUT_DIMENSIONS));

		BiasMatrix layer1Biases = new BiasMatrixImpl(
				matrixFactory.createMatrixFromRowsByRowsArray(6, 1, helper.deserialize(float[].class, "layer1Biases")));

		WeightsMatrix layer3Weights = new WeightsMatrixImpl(
				matrixFactory.createMatrixFromRowsByRowsArray(400, 600,
						helper.deserialize(float[].class, "layer3Weights")),
				new WeightsFormatImpl(Arrays.asList(Dimension.INPUT_FEATURE), Arrays.asList(Dimension.OUTPUT_FEATURE), WeightsMatrixOrientation.ROWS_SPAN_OUTPUT_DIMENSIONS));

		BiasMatrix layer3Biases = new BiasMatrixImpl(matrixFactory.createMatrixFromRowsByRowsArray(400, 1,
				helper.deserialize(float[].class, "layer3Biases")));

		WeightsMatrix layer4Weights = new WeightsMatrixImpl(
				matrixFactory.createMatrixFromRowsByRowsArray(100, 400,
						helper.deserialize(float[].class, "layer4Weights")),
				new WeightsFormatImpl(Arrays.asList(Dimension.INPUT_FEATURE), Arrays.asList(Dimension.OUTPUT_FEATURE), 
						WeightsMatrixOrientation.ROWS_SPAN_OUTPUT_DIMENSIONS));

		BiasMatrix layer4Biases = new BiasMatrixImpl(matrixFactory.createMatrixFromRowsByRowsArray(100, 1,
				helper.deserialize(float[].class, "layer4Biases")));

		WeightsMatrix layer5Weights = new WeightsMatrixImpl(
				matrixFactory.createMatrixFromRowsByRowsArray(10, 100,
						helper.deserialize(float[].class, "layer5Weights")),
				new WeightsFormatImpl(Arrays.asList(Dimension.INPUT_FEATURE), Arrays.asList(Dimension.OUTPUT_FEATURE), WeightsMatrixOrientation.ROWS_SPAN_OUTPUT_DIMENSIONS));

		BiasMatrix layer5Biases = new BiasMatrixImpl(matrixFactory.createMatrixFromRowsByRowsArray(10, 1,
				helper.deserialize(float[].class, "layer5Biases")));

    // Construct a Neural Network in the same shape as our Kaggle entry.
    // Initialise each trainable layer with our pre-trained weights.
		
	SupervisedFeedForwardNeuralNetworkFactory neuralNetworkFactory = 
			new DefaultSupervisedFeedForwardNeuralNetworkFactory(directedComponentFactory);
	
	LayeredSupervisedFeedForwardNeuralNetworkFactory layredNeuralNetworkFactory = 
			new DefaultLayeredSupervisedFeedForwardNeuralNetworkFactory(directedComponentFactory);
		
	DefaultSessionFactory sessionFactory = 	new DefaultSessionFactoryImpl(matrixFactory, directedComponentFactory, layerFactory, neuralNetworkFactory, layredNeuralNetworkFactory);	
    
	return sessionFactory.createSession().buildNeuralNetwork("mnist")
		.withConvolutionalLayer("FirstLayer")
			.withInputNeurons(new Neurons3D(28, 28 ,1, true))
			.withWeightsMatrix(layer1Weights)
			.withBiasMatrix(layer1Biases)
			.withOutputNeurons(new Neurons3D(20, 20, 6, false))
			.withActivationFunction(ActivationFunctionBaseType.SIGMOID)
		.withMaxPoolingLayer("SecondLayer")
		.withInputNeurons(new Neurons3D(20, 20, 6, false))
		//  The max pooling layer that this NN was trained with originally was a legacy
		//  implementation which scaled up the output activations by a factor of
		//  4.  This scaleOutputs configuration is usually not required   
		.withScaleOutputs()
		.withConfig(config -> config.withStrideHeight(2).withStrideWidth(2)
				.withOutputNeurons(new Neurons3D(10, 10, 6, false)))
		.withFullyConnectedLayer("ThirdLayer")
			.withInputNeurons(new Neurons3D(10, 10, 6, true))
			.withWeightsMatrix(layer3Weights)
			.withBiasMatrix(layer3Biases)
			.withOutputNeurons(new Neurons3D(5, 5, 16, false))
			.withActivationFunction(ActivationFunctionBaseType.SIGMOID)
		.withFullyConnectedLayer("FourthLayer")
			.withInputNeurons(new Neurons(400, true))
			.withWeightsMatrix(layer4Weights)
			.withBiasMatrix(layer4Biases)
			.withOutputNeurons(new Neurons(100, false))
			.withActivationFunction(ActivationFunctionBaseType.SIGMOID)
		.withFullyConnectedLayer("FifthLayer")
			.withInputNeurons(new Neurons(100, true))
			.withWeightsMatrix(layer5Weights)
			.withBiasMatrix(layer5Biases)
			.withOutputNeurons(new Neurons(10, false))
			.withActivationFunction(ActivationFunctionBaseType.SOFTMAX)
		.buildLayeredNeuralNetwork();

  }

  @Override
  protected NeuronsActivation createTrainingDataNeuronActivations(MatrixFactory matrixFactory) {
    LOGGER.trace("Creating training data NeuronsActivation");
    
    DoubleArrayMatrixLoader loader = new DoubleArrayMatrixLoader(
            PretrainedKaggleCompetionClassifierDemo.class.getClassLoader());
    // Load Mnist data into double[][] matrices
    float[][] trainingDataMatrix = toFloatArray(loader.loadDoubleMatrixFromCsv("train.csv",
            new KagglePixelFeaturesMatrixCsvDataExtractor(), 1, 1001));
    
    NeuronsActivation act = new NeuronsActivationImpl(new Neurons(trainingDataMatrix[0].length, false), matrixFactory.createMatrixFromRows(trainingDataMatrix).transpose(),
    		ImageNeuronsActivationFormat.ML4J_DEFAULT_IMAGE_FORMAT, true);
    
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

    return new NeuronsActivationImpl(new Neurons(testDataMatrix[0].length, false),  matrixFactory.createMatrixFromRows(testDataMatrix).transpose(),
    		ImageNeuronsActivationFormat.ML4J_DEFAULT_IMAGE_FORMAT, true);
  }

  @Override
  protected MatrixFactory createMatrixFactory() {
    LOGGER.trace("Creating MatrixFactory");
    return new JBlasRowMajorMatrixFactoryOptimised();
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
      
      NeuronsActivation orignalActivation = new NeuronsActivationImpl(new Neurons(activations.getRows(), false), activations,
    		  testDataInputActivations.getFormat());

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
   
    return new NeuronsActivationImpl(new Neurons(testDataMatrix[0].length, false), matrixFactory.createMatrixFromRows(testDataMatrix).transpose(),
    		NeuronsActivationFormat.ROWS_SPAN_FEATURE_SET, true);
  }

  @Override
  protected NeuronsActivation createTestSetLabelNeuronActivations(MatrixFactory matrixFactory) {
    DoubleArrayMatrixLoader loader = new DoubleArrayMatrixLoader(
        PretrainedKaggleCompetionClassifierDemo.class.getClassLoader());
    // Load Mnist data into double[][] matrices
    float[][] testDataMatrix = toFloatArray(loader.loadDoubleMatrixFromCsv("train.csv",
        new SingleDigitLabelsMatrixCsvDataExtractor(), 1001, 2001));
   
    return new NeuronsActivationImpl(new Neurons(testDataMatrix[0].length, false), matrixFactory.createMatrixFromRows(testDataMatrix).transpose(),
    		NeuronsActivationFormat.ROWS_SPAN_FEATURE_SET, true);
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
