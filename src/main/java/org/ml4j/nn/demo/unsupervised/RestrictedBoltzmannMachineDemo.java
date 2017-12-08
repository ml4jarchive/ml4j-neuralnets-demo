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
import org.ml4j.nn.axons.FullyConnectedAxons;
import org.ml4j.nn.demo.base.unsupervised.UnsupervisedNeuralNetworkDemoBase;
import org.ml4j.nn.demo.util.MnistUtils;
import org.ml4j.nn.demo.util.PixelFeaturesMatrixCsvDataExtractor;
import org.ml4j.nn.layers.RestrictedBoltzmannLayer;
import org.ml4j.nn.layers.RestrictedBoltzmannLayerImpl;
import org.ml4j.nn.layers.UndirectedLayerContext;
import org.ml4j.nn.layers.UndirectedLayerContextImpl;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.Neurons3D;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.neurons.NeuronsActivationFeatureOrientation;
import org.ml4j.nn.unsupervised.RestrictedBoltzmannMachine;
import org.ml4j.nn.unsupervised.RestrictedBoltzmannMachineContext;
import org.ml4j.nn.unsupervised.RestrictedBoltzmannMachineContextImpl;
import org.ml4j.nn.unsupervised.RestrictedBoltzmannMachineImpl;
import org.ml4j.util.DoubleArrayMatrixLoader;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * A simple test harness to train and showcase a RestrictedBoltzmannMachine.
 * 
 * @author Michael Lavelle
 *
 */
public class RestrictedBoltzmannMachineDemo
    extends UnsupervisedNeuralNetworkDemoBase<RestrictedBoltzmannMachine, 
    RestrictedBoltzmannMachineContext> {
  
  private double learningRate = 0.01;

  private static final Logger LOGGER = 
        LoggerFactory.getLogger(RestrictedBoltzmannMachineDemo.class);

  public static void main(String[] args) throws Exception {
    RestrictedBoltzmannMachineDemo demo = new RestrictedBoltzmannMachineDemo();
    demo.runDemo();
  }

  @Override
  protected RestrictedBoltzmannMachine createUnsupervisedNeuralNetwork(int featureCount) {

    // Construct a RestrictedBoltzmannMachine
    
    MatrixFactory matrixFactory = createMatrixFactory();
    
    NeuronsActivation trainingDataActivations = createTrainingDataNeuronActivations(matrixFactory);
    
    Matrix initialConnectionWeights = 
        RestrictedBoltzmannLayerImpl
        .generateInitialConnectionWeights(trainingDataActivations, 
            new Neurons3D(28, 28 ,1, true), new Neurons(100, true), learningRate, matrixFactory);
        
    RestrictedBoltzmannLayer<FullyConnectedAxons> restrictedBoltmannLayer 
        = new RestrictedBoltzmannLayerImpl(
        new Neurons3D(28, 28 ,1, true), new Neurons(100, true), 
        new SigmoidActivationFunction(), new SigmoidActivationFunction(), matrixFactory, 
        initialConnectionWeights);
    
    return new RestrictedBoltzmannMachineImpl(restrictedBoltmannLayer);
  }

  @Override
  protected NeuronsActivation createTrainingDataNeuronActivations(MatrixFactory matrixFactory) {
    LOGGER.trace("Creating training data NeuronsActivation");
    
    DoubleArrayMatrixLoader loader = new DoubleArrayMatrixLoader(
            RestrictedBoltzmannMachineDemo.class.getClassLoader());
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
        RestrictedBoltzmannMachineDemo.class.getClassLoader());
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
  protected RestrictedBoltzmannMachineContext createTrainingContext(
      RestrictedBoltzmannMachine unsupervisedNeuralNetwork,
      MatrixFactory matrixFactory) {
    LOGGER.trace("Creating RestrictedBoltzmannMachineContext");
    RestrictedBoltzmannMachineContext context = 
        new RestrictedBoltzmannMachineContextImpl(matrixFactory);
    context.setTrainingEpochs(200);
    context.setTrainingLearningRate(learningRate);
    context.setTrainingMiniBatchSize(32);
    return context;
  }

  @Override
  protected void showcaseTrainedNeuralNetwork(RestrictedBoltzmannMachine restrictedBoltzmannMachine,
      NeuronsActivation testDataInputActivations, MatrixFactory matrixFactory) throws Exception {
    LOGGER.info("Showcasing trained RestrictedBoltzmannMachine...");
    
    // Create display for our demo
    ImageDisplay<Long> display = new ImageDisplay<Long>(280, 280);
    
    LOGGER.info("Drawing visualisations of patterns sought by the hidden neurons...");
    for (int j = 0; j < restrictedBoltzmannMachine.getFirstLayer()
        .getHiddenNeurons().getNeuronCountExcludingBias(); j++) {
      UndirectedLayerContext hiddenNeuronInspectionContext =
          new UndirectedLayerContextImpl(0, matrixFactory);
      NeuronsActivation neuronActivationMaximisingActivation =
          restrictedBoltzmannMachine.getFirstLayer().getOptimalVisibleActivationsForHiddenNeuron(j,
              hiddenNeuronInspectionContext);
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
  }
}
