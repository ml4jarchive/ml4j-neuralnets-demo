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
import org.ml4j.jblas.JBlasMatrixFactory;
import org.ml4j.nn.activationfunctions.SigmoidActivationFunction;
import org.ml4j.nn.demo.base.unsupervised.UnsupervisedNeuralNetworkDemoBase;
import org.ml4j.nn.demo.util.MnistUtils;
import org.ml4j.nn.demo.util.PixelFeaturesMatrixCsvDataExtractor;
import org.ml4j.nn.layers.DirectedLayerContext;
import org.ml4j.nn.layers.FeedForwardLayer;
import org.ml4j.nn.layers.FullyConnectedFeedForwardLayerImpl;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.Neurons3D;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.neurons.NeuronsActivationFeatureOrientation;
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
    
    FeedForwardLayer<?, ?> encodingLayer = new FullyConnectedFeedForwardLayerImpl(
        new Neurons3D(28, 28 ,1, true), new Neurons(200, false), 
        new SigmoidActivationFunction(), matrixFactory, false);
    
    FeedForwardLayer<?, ?> decodingLayer = 
        new FullyConnectedFeedForwardLayerImpl(new Neurons(200, true), 
        new Neurons3D(28, 28 ,1, false), new SigmoidActivationFunction(), matrixFactory, false);

    return new AutoEncoderImpl(encodingLayer, decodingLayer);
  }

  @Override
  protected NeuronsActivation createTrainingDataNeuronActivations(MatrixFactory matrixFactory) {
    LOGGER.trace("Creating training data NeuronsActivation");
    
    DoubleArrayMatrixLoader loader = new DoubleArrayMatrixLoader(
            AutoEncoderDemo.class.getClassLoader());
    // Load Mnist data into double[][] matrices
    double[][] trainingDataMatrix = loader.loadDoubleMatrixFromCsv("mnist2500_X_custom.csv",
            new PixelFeaturesMatrixCsvDataExtractor(), 0, 500);
    
    return new NeuronsActivation(matrixFactory.createMatrix(trainingDataMatrix),
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

    return new NeuronsActivation(matrixFactory.createMatrix(testDataMatrix),
        NeuronsActivationFeatureOrientation.COLUMNS_SPAN_FEATURE_SET);
  }

  @Override
  protected MatrixFactory createMatrixFactory() {
    LOGGER.trace("Creating MatrixFactory");
    return new JBlasMatrixFactory();
  }

  @Override
  protected AutoEncoderContext createTrainingContext(AutoEncoder unsupervisedNeuralNetwork,
      MatrixFactory matrixFactory) {
    LOGGER.trace("Creating AutoEncoderContext");
    // Train from layer index 0 to the end layer
    AutoEncoderContext context = new AutoEncoderContextImpl(matrixFactory, 0, null);
    context.setTrainingEpochs(400);
    context.setTrainingLearningRate(0.1);
    return context;
  }

  @Override
  protected void showcaseTrainedNeuralNetwork(AutoEncoder autoEncoder,
      NeuronsActivation testDataInputActivations, MatrixFactory matrixFactory) throws Exception {
    LOGGER.info("Showcasing trained AutoEncoder...");

    // Create display for our demo
    ImageDisplay<Long> display = new ImageDisplay<Long>(280, 280);
    
    // Create a context for the first layer only
    AutoEncoderContext autoEncoderNeuronVisualisationContext =  
        new AutoEncoderContextImpl(matrixFactory, 0, 0);
    
    DirectedLayerContext hiddenNeuronInspectionContext = 
        autoEncoderNeuronVisualisationContext.getLayerContext(0);
    
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
      
      NeuronsActivation orignalActivation = new NeuronsActivation(activations,
          NeuronsActivationFeatureOrientation.COLUMNS_SPAN_FEATURE_SET);

      MnistUtils.draw(orignalActivation.getActivations().toArray(), display);

      // Encode only through the first layer
      AutoEncoderContext encodingContext = new AutoEncoderContextImpl(matrixFactory, 0, 0);

      NeuronsActivation encodedFeatures = autoEncoder.encode(orignalActivation, encodingContext);

      LOGGER.info("Encoded a single image from " + orignalActivation.getFeatureCount() 
          + " pixels to " + encodedFeatures.getFeatureCount() + " features");
      
      Thread.sleep(1000);

      // Decode only through the seconds layer
      AutoEncoderContext decodingContext = new AutoEncoderContextImpl(matrixFactory, 1, 1);

      // Now reconstruct the features again
      NeuronsActivation reconstructedFeatures =
          autoEncoder.decode(encodedFeatures, decodingContext);
      
      // Display the reconstructed input image
      MnistUtils.draw(reconstructedFeatures.getActivations().toArray(), display);
      LOGGER.info("Decoded " + encodedFeatures.getFeatureCount() 
          + " features into an image with "  + reconstructedFeatures.getFeatureCount() 
          + " pixels");

      Thread.sleep(1000);
    }
  }
}
