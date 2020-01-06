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

package org.ml4j.nn.demo.unsupervised;

import org.ml4j.Matrix;
import org.ml4j.MatrixFactory;
import org.ml4j.imaging.targets.ImageDisplay;
import org.ml4j.jblas.JBlasRowMajorMatrixFactory;
import org.ml4j.nn.activationfunctions.DefaultSigmoidActivationFunctionImpl;
import org.ml4j.nn.axons.factories.AxonsFactory;
import org.ml4j.nn.components.factories.DirectedComponentFactory;
import org.ml4j.nn.demo.base.unsupervised.UnsupervisedNeuralNetworkDemoBase;
import org.ml4j.nn.demo.util.MnistUtils;
import org.ml4j.nn.demo.util.PixelFeaturesMatrixCsvDataExtractor;
import org.ml4j.nn.factories.DefaultAxonsFactoryImpl;
import org.ml4j.nn.factories.DefaultDirectedComponentFactoryImpl;
import org.ml4j.nn.layers.DirectedLayerContext;
import org.ml4j.nn.layers.FeedForwardLayer;
import org.ml4j.nn.layers.FullyConnectedFeedForwardLayerImpl;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.Neurons3D;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.neurons.NeuronsActivationFeatureOrientation;
import org.ml4j.nn.neurons.NeuronsActivationImpl;
import org.ml4j.nn.unsupervised.AutoEncoder;
import org.ml4j.nn.unsupervised.AutoEncoderContext;
import org.ml4j.nn.unsupervised.AutoEncoderContextImpl;
import org.ml4j.nn.unsupervised.AutoEncoderImpl;
import org.ml4j.util.DoubleArrayMatrixLoader;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * A simple test harness to train and showcase an AutoEncoder.
 * 
 * @author Michael Lavelle
 *
 */
public class AutoEncoderDemo
    extends UnsupervisedNeuralNetworkDemoBase<AutoEncoder, AutoEncoderContext> {

  private static final Logger LOGGER = LoggerFactory.getLogger(AutoEncoderDemo.class);

  public static void main(String[] args) throws Exception {
    AutoEncoderDemo demo = new AutoEncoderDemo();
    demo.runDemo();
  }

  @Override
  protected AutoEncoder createUnsupervisedNeuralNetwork(int featureCount) {

    // Construct a 2 layer AutoEncoder
    
    MatrixFactory matrixFactory = createMatrixFactory();
    
    AxonsFactory axonsFactory = new DefaultAxonsFactoryImpl(matrixFactory);
    
    DirectedComponentFactory directedComponentFactory = new DefaultDirectedComponentFactoryImpl(matrixFactory, axonsFactory);
    
    FeedForwardLayer<?, ?> encodingLayer = new FullyConnectedFeedForwardLayerImpl(directedComponentFactory, 
        axonsFactory, new Neurons3D(28, 28 ,1, true), new Neurons(200, false), 
        new DefaultSigmoidActivationFunctionImpl(), matrixFactory, false);
    
    FeedForwardLayer<?, ?> decodingLayer = 
        new FullyConnectedFeedForwardLayerImpl(directedComponentFactory, axonsFactory, new Neurons(200, true), 
        new Neurons3D(28, 28 ,1, false), new DefaultSigmoidActivationFunctionImpl(), matrixFactory, false);

    return new AutoEncoderImpl(directedComponentFactory, encodingLayer, decodingLayer);
  }

  @Override
  protected NeuronsActivation createTrainingDataNeuronActivations(MatrixFactory matrixFactory) {
    LOGGER.trace("Creating training data NeuronsActivation");
    
    DoubleArrayMatrixLoader loader = new DoubleArrayMatrixLoader(
            AutoEncoderDemo.class.getClassLoader());
    // Load Mnist data into double[][] matrices
    float[][] trainingDataMatrix = toFloatArray(loader.loadDoubleMatrixFromCsv("mnist2500_X_custom.csv",
            new PixelFeaturesMatrixCsvDataExtractor(), 0, 500));
    
    return new NeuronsActivationImpl(matrixFactory.createMatrixFromRows(trainingDataMatrix).transpose(),
        NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET);
  }

  @Override
  protected NeuronsActivation createTestSetDataNeuronActivations(MatrixFactory matrixFactory) {
    LOGGER.trace("Creating test data NeuronsActivation");
    
    DoubleArrayMatrixLoader loader = new DoubleArrayMatrixLoader(
        AutoEncoderDemo.class.getClassLoader());
    // Load Mnist data into double[][] matrices
    float[][] testDataMatrix = toFloatArray(loader.loadDoubleMatrixFromCsv("mnist2500_X_custom.csv",
        new PixelFeaturesMatrixCsvDataExtractor(), 1000, 2000));

    return new NeuronsActivationImpl(matrixFactory.createMatrixFromRows(testDataMatrix).transpose(),
        NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET);
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
  protected MatrixFactory createMatrixFactory() {
    LOGGER.trace("Creating MatrixFactory");
    return new JBlasRowMajorMatrixFactory();
  }

  @Override
  protected AutoEncoderContext createTrainingContext(AutoEncoder unsupervisedNeuralNetwork,
      MatrixFactory matrixFactory) {
    LOGGER.trace("Creating AutoEncoderContext");
    // Train from layer index 0 to the end layer
    AutoEncoderContext context = new AutoEncoderContextImpl(matrixFactory, 0, null, true);
    context.setTrainingEpochs(400);
    context.setTrainingLearningRate(0.1f);
    return context;
  }

  @Override
  protected void showcaseTrainedNeuralNetwork(AutoEncoder autoEncoder,
      NeuronsActivation testDataInputActivations, MatrixFactory matrixFactory) throws Exception {
    LOGGER.info("Showcasing trained AutoEncoder...");

    // Create display for our demo
    ImageDisplay<Long> display = new ImageDisplay<>(280, 280);
    
    // Create a context for the first layer only
    AutoEncoderContext autoEncoderNeuronVisualisationContext =  
        new AutoEncoderContextImpl(matrixFactory, 0, 0, false);
    
    DirectedLayerContext hiddenNeuronInspectionContext = 
        autoEncoderNeuronVisualisationContext.getLayerContext(0);
    
    LOGGER.info("Drawing visualisations of patterns sought by the hidden neurons...");
    for (int j = 0; j < autoEncoder.getFirstLayer().getOutputNeuronCount(); j++) {
      NeuronsActivation neuronActivationMaximisingActivation = autoEncoder.getFirstLayer()
          .getOptimalInputForOutputNeuron(j, hiddenNeuronInspectionContext);
      float[] neuronActivationMaximisingFeatures =
          neuronActivationMaximisingActivation.getActivations(matrixFactory).getRowByRowArray();

      float[] intensities = new float[neuronActivationMaximisingFeatures.length];
      for (int i = 0; i < intensities.length; i++) {
        double val = neuronActivationMaximisingFeatures[i];
        float boundary = 0.02f;
        intensities[i] = val < -boundary ? 0f : val > boundary ? 1f : 0.5f;
      }
      MnistUtils.draw(intensities, display);
      Thread.sleep(100);
    }
    
    // Visualise the reconstructions of the input data
  
    LOGGER.info("Visualising reconstructed data");
    for (int i = 0; i < 100; i++) {

      // For each element in our test set, obtain the compressed encoded features
      Matrix activations = testDataInputActivations.getActivations(matrixFactory).getColumn(i);
      
      NeuronsActivation orignalActivation = new NeuronsActivationImpl(activations,
          NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET);

      MnistUtils.draw(orignalActivation.getActivations(matrixFactory).getRowByRowArray(), display);

      // Encode only through the first layer
      AutoEncoderContext encodingContext = new AutoEncoderContextImpl(matrixFactory, 0, 0, false);

      NeuronsActivation encodedFeatures = autoEncoder.encode(orignalActivation, encodingContext);

      LOGGER.info("Encoded a single image from " + orignalActivation.getFeatureCount() 
          + " pixels to " + encodedFeatures.getFeatureCount() + " features");
      
      Thread.sleep(1000);

      // Decode only through the second layer
      AutoEncoderContext decodingContext = new AutoEncoderContextImpl(matrixFactory, 1, 1, false);

      // Now reconstruct the features again
      NeuronsActivation reconstructedFeatures =
          autoEncoder.decode(encodedFeatures, decodingContext);
      
      // Display the reconstructed input image
      MnistUtils.draw(reconstructedFeatures.getActivations(matrixFactory).getRowByRowArray(), display);
      LOGGER.info("Decoded " + encodedFeatures.getFeatureCount() 
          + " features into an image with "  + reconstructedFeatures.getFeatureCount() 
          + " pixels");

      Thread.sleep(1000);
    }
  }
}
