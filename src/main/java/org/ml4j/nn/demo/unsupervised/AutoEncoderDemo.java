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
import org.ml4j.layers.mocks.FeedForwardLayerMock;
import org.ml4j.mocks.MatrixFactoryMock;
import org.ml4j.nn.demo.base.unsupervised.UnsupervisedNeuralNetworkDemoBase;
import org.ml4j.nn.demo.util.MnistUtils;
import org.ml4j.nn.demo.util.PixelFeaturesMatrixCsvDataExtractor;
import org.ml4j.nn.layers.DirectedLayerContext;
import org.ml4j.nn.layers.FeedForwardLayer;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.neurons.NeuronsActivationFeatureOrientation;
import org.ml4j.nn.unsupervised.AutoEncoder;
import org.ml4j.nn.unsupervised.AutoEncoderContext;
import org.ml4j.nn.unsupervised.mocks.AutoEncoderContextMock;
import org.ml4j.nn.unsupervised.mocks.AutoEncoderMock;
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
  protected AutoEncoder createUnsupervisedNeuralNetwork(int featureCount,
      boolean isBiasUnitIncluded) {

    // Construct a 2 layer AutoEncoderMock

    FeedForwardLayer<?> encodingLayer = new FeedForwardLayerMock(784, 100);
    FeedForwardLayer<?> decodingLayer = new FeedForwardLayerMock(100, 784);

    return new AutoEncoderMock(encodingLayer, decodingLayer);
  }

  @Override
  protected NeuronsActivation createTrainingDataNeuronActivations(MatrixFactory matrixFactory) {
    LOGGER.trace("Creating training data NeuronsActivation");
    
    DoubleArrayMatrixLoader loader = new DoubleArrayMatrixLoader(
            AutoEncoderDemo.class.getClassLoader());
    // Load Mnist data into double[][] matrices
    double[][] trainingDataMatrix = loader.loadDoubleMatrixFromCsv("mnist2500_X_custom.csv",
            new PixelFeaturesMatrixCsvDataExtractor(), 0, 1000);
    
    return new NeuronsActivation(matrixFactory.createMatrix(trainingDataMatrix), false,
        NeuronsActivationFeatureOrientation.COLUMNS_SPAN_FEATURE_SET);
  }

  @Override
  protected NeuronsActivation createTestSetDataNeuronActivations(MatrixFactory matrixFactory) {
    LOGGER.trace("Creating test data NeuronsActivation");
    
    DoubleArrayMatrixLoader loader = new DoubleArrayMatrixLoader(
        AutoEncoderDemo.class.getClassLoader());
    // Load Mnist data into double[][] matrices
    double[][] testDataMatrix = loader.loadDoubleMatrixFromCsv("mnist2500_X_custom.csv",
        new PixelFeaturesMatrixCsvDataExtractor(), 1000, 2000);

    return new NeuronsActivation(matrixFactory.createMatrix(testDataMatrix), false,
        NeuronsActivationFeatureOrientation.COLUMNS_SPAN_FEATURE_SET);
  }

  @Override
  protected MatrixFactory createMatrixFactory() {
    LOGGER.trace("Creating MatrixFactoryMock");
    return new MatrixFactoryMock();
  }

  @Override
  protected AutoEncoderContext createTrainingContext(AutoEncoder unsupervisedNeuralNetwork,
      MatrixFactory matrixFactory) {
    LOGGER.trace("Creating AutoEncoderContextMock");
    return new AutoEncoderContextMock(matrixFactory);
  }

  @Override
  protected void showcaseTrainedNeuralNetwork(AutoEncoder autoEncoder,
      NeuronsActivation testDataInputActivations, MatrixFactory matrixFactory) throws Exception {
    LOGGER.info("Showcasing trained AutoEncoder...");

    // Create display for our demo
    ImageDisplay<Long> display = new ImageDisplay<Long>(280, 280);
    
    AutoEncoderContext autoEncoderContext =  new AutoEncoderContextMock(matrixFactory);
    
    DirectedLayerContext hiddenNeuronInspectionContext = autoEncoderContext.createLayerContext(0);
    
    LOGGER.info("Drawing visualisations of patterns sought by the hidden neurons...");
    for (int j = 0; j < autoEncoder.getFirstLayer().getOutputNeuronCount(); j++) {
      NeuronsActivation neuronActivationMaximisingActivation = autoEncoder.getFirstLayer()
          .getOptimalInputForOutputNeuron(j, hiddenNeuronInspectionContext);
      double[] neuronActivationMaximisingFeatures =
          neuronActivationMaximisingActivation.getActivations().toArray();

      double[] intensities = new double[neuronActivationMaximisingFeatures.length];
      for (int i = 0; i < intensities.length; i++) {
        double val = neuronActivationMaximisingFeatures[i];
        double boundary = 0.02;
        intensities[i] = val < -boundary ? 0 : val > boundary ? 1 : 0.5;
      }
      MnistUtils.draw(intensities, display);
      Thread.sleep(100);
    }
    
    // Visualise the reconstructions of the input data
  
    LOGGER.info("Visualising reconstructed data");
    for (int i = 0; i < 100; i++) {

      // For each element in our test set, obtain the compressed encoded features
      Matrix activations = testDataInputActivations.getActivations().getRow(i);
      
      NeuronsActivation orignalActivation = new NeuronsActivation(activations, false,
          NeuronsActivationFeatureOrientation.COLUMNS_SPAN_FEATURE_SET);

      MnistUtils.draw(orignalActivation.getActivations().toArray(), display);

      AutoEncoderContext encodingContext = new AutoEncoderContextMock(matrixFactory);

      NeuronsActivation encodedFeatures = autoEncoder.encode(orignalActivation, encodingContext);

      LOGGER.info("Encoded a single image from " + orignalActivation.getFeatureCountExcludingBias() 
          + " pixels to " + encodedFeatures.getFeatureCountExcludingBias() + " features");
      
      Thread.sleep(1000);

      AutoEncoderContext decodingContext = new AutoEncoderContextMock(matrixFactory);

      // Now reconstruct the features again
      NeuronsActivation reconstructedFeatures =
          autoEncoder.decode(encodedFeatures, decodingContext);
      
      // Display the reconstructed input image
      MnistUtils.draw(reconstructedFeatures.getActivations().toArray(), display);
      LOGGER.info("Decoded " + encodedFeatures.getFeatureCountExcludingBias() 
          + " features into an image with "  + reconstructedFeatures.getFeatureCountExcludingBias() 
          + " pixels");

      Thread.sleep(1000);
    }
  }
}
