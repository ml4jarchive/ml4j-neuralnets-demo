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

import org.ml4j.MatrixFactory;
import org.ml4j.jblas.JBlasMatrixFactory;
import org.ml4j.nn.activationfunctions.SigmoidActivationFunction;
import org.ml4j.nn.axons.FullyConnectedAxons;
import org.ml4j.nn.demo.base.unsupervised.UnsupervisedNeuralNetworkDemoBase;
import org.ml4j.nn.demo.util.PixelFeaturesMatrixCsvDataExtractor;
import org.ml4j.nn.layers.RestrictedBoltzmannLayer;
import org.ml4j.nn.layers.RestrictedBoltzmannLayerImpl;
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
    
    RestrictedBoltzmannLayer<FullyConnectedAxons> restrictedBoltmannLayer 
        = new RestrictedBoltzmannLayerImpl(
        new Neurons3D(28, 28 ,1, true), new Neurons3D(28, 28 ,1, true), 
        new SigmoidActivationFunction(), new SigmoidActivationFunction(), matrixFactory);
    
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
    return context;
  }

  @Override
  protected void showcaseTrainedNeuralNetwork(RestrictedBoltzmannMachine restrictedBoltzmannMachine,
      NeuronsActivation testDataInputActivations, MatrixFactory matrixFactory) throws Exception {
    LOGGER.info("Showcasing trained RestrictedBoltzmannMachine...");
  }
}
