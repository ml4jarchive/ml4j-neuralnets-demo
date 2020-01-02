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
import org.ml4j.nn.activationfunctions.SigmoidActivationFunction;
import org.ml4j.nn.axons.TrainableAxons;
import org.ml4j.nn.axons.factories.AxonsFactory;
import org.ml4j.nn.demo.base.unsupervised.UnsupervisedNeuralNetworkDemoBase;
import org.ml4j.nn.demo.util.MnistUtils;
import org.ml4j.nn.demo.util.PixelFeaturesMatrixCsvDataExtractor;
import org.ml4j.nn.factories.PrototypeAxonsFactoryImpl;
import org.ml4j.nn.layers.RestrictedBoltzmannLayer;
import org.ml4j.nn.layers.RestrictedBoltzmannLayerImpl;
import org.ml4j.nn.layers.UndirectedLayerContext;
import org.ml4j.nn.layers.UndirectedLayerContextImpl;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.Neurons3D;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.neurons.NeuronsActivationFeatureOrientation;
import org.ml4j.nn.neurons.NeuronsActivationImpl;
import org.ml4j.nn.unsupervised.RestrictedBoltzmannMachine;
import org.ml4j.nn.unsupervised.RestrictedBoltzmannMachineContext;
import org.ml4j.nn.unsupervised.RestrictedBoltzmannMachineContextImpl;
import org.ml4j.nn.unsupervised.RestrictedBoltzmannMachineImpl;
import org.ml4j.nn.unsupervised.RestrictedBoltzmannSamplingActivation;
import org.ml4j.util.DoubleArrayMatrixLoader;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * A simple test harness to train and showcase a RestrictedBoltzmannMachine.
 * 
 * @author Michael Lavelle
 *
 */
public class RestrictedBoltzmannMachineDemo extends
    UnsupervisedNeuralNetworkDemoBase<RestrictedBoltzmannMachine, 
    RestrictedBoltzmannMachineContext> {

  private float learningRate = 0.01f;

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
    
    AxonsFactory axonsFactory = new PrototypeAxonsFactoryImpl(matrixFactory);

    NeuronsActivation trainingDataActivations = createTrainingDataNeuronActivations(matrixFactory);

    Matrix initialConnectionWeightsAndBiases =
        RestrictedBoltzmannLayerImpl.generateInitialConnectionWeights(trainingDataActivations,
            new Neurons3D(28, 28, 1, true), new Neurons(100, true), learningRate, matrixFactory);
    
    int[] inds = new int[initialConnectionWeightsAndBiases.getRows() - 1];
    for (int i = 0; i < inds.length; i++) {
    	inds[i] = i + 1;
    }
    int[] inds2 = new int[initialConnectionWeightsAndBiases.getColumns() - 1];
    for (int i = 0; i < inds2.length; i++) {
    	inds2[i] = i + 1;
    }
    Matrix initialConnectionWeights = initialConnectionWeightsAndBiases.getRows(inds).getColumns(inds2).transpose();
    Matrix initialLeftToRightBiases = initialConnectionWeightsAndBiases.getRow(0).getColumns(inds2).transpose();
    Matrix initialRightToLeftBiases = initialConnectionWeightsAndBiases.getColumn(0).getRows(inds).transpose();

    RestrictedBoltzmannLayer<TrainableAxons<?, ?, ?>> restrictedBoltmannLayer =
        new RestrictedBoltzmannLayerImpl(axonsFactory, new Neurons3D(28, 28, 1, true), new Neurons(100, true),
            new SigmoidActivationFunction(), new SigmoidActivationFunction(), matrixFactory,
            initialConnectionWeights, initialLeftToRightBiases, initialRightToLeftBiases.transpose());

    return new RestrictedBoltzmannMachineImpl(restrictedBoltmannLayer);
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
  protected NeuronsActivation createTrainingDataNeuronActivations(MatrixFactory matrixFactory) {
    LOGGER.trace("Creating training data NeuronsActivation");

    DoubleArrayMatrixLoader loader =
        new DoubleArrayMatrixLoader(RestrictedBoltzmannMachineDemo.class.getClassLoader());
    // Load Mnist data into double[][] matrices
    float[][] trainingDataMatrix = toFloatArray(loader.loadDoubleMatrixFromCsv("mnist2500_X_custom.csv",
        new PixelFeaturesMatrixCsvDataExtractor(), 0, 1000));

    return new NeuronsActivationImpl(matrixFactory.createMatrixFromRows(trainingDataMatrix).transpose(),
        NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET);
  }

  @Override
  protected NeuronsActivation createTestSetDataNeuronActivations(MatrixFactory matrixFactory) {
    LOGGER.trace("Creating test data NeuronsActivation");

    DoubleArrayMatrixLoader loader =
        new DoubleArrayMatrixLoader(RestrictedBoltzmannMachineDemo.class.getClassLoader());
    // Load Mnist data into double[][] matrices
    float[][] testDataMatrix = toFloatArray(loader.loadDoubleMatrixFromCsv("mnist2500_X_custom.csv",
        new PixelFeaturesMatrixCsvDataExtractor(), 1000, 2000));

    return new NeuronsActivationImpl(matrixFactory.createMatrixFromRows(testDataMatrix).transpose(),
        NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET);
  }

  @Override
  protected MatrixFactory createMatrixFactory() {
    LOGGER.trace("Creating MatrixFactory");
    return new JBlasRowMajorMatrixFactory();
  }

  @Override
  protected RestrictedBoltzmannMachineContext createTrainingContext(
      RestrictedBoltzmannMachine unsupervisedNeuralNetwork, MatrixFactory matrixFactory) {
    LOGGER.trace("Creating RestrictedBoltzmannMachineContext");
    RestrictedBoltzmannMachineContext context =
        new RestrictedBoltzmannMachineContextImpl(matrixFactory, true);
    context.setTrainingEpochs(300);
    context.setTrainingLearningRate(learningRate);
    //context.setTrainingMiniBatchSize(64);
    return context;
  }

  @Override
  protected void showcaseTrainedNeuralNetwork(RestrictedBoltzmannMachine restrictedBoltzmannMachine,
      NeuronsActivation testDataInputActivations, MatrixFactory matrixFactory) throws Exception {
    LOGGER.info("Showcasing trained RestrictedBoltzmannMachine...");

    // Create display for our demo
    ImageDisplay<Long> display = new ImageDisplay<>(280, 280);


    RestrictedBoltzmannMachineContext samplingContext =
        new RestrictedBoltzmannMachineContextImpl(matrixFactory, false);
    
    LOGGER.info("Drawing visualisations of patterns sought by the hidden neurons...");
    for (int j = 0; j < restrictedBoltzmannMachine.getLayer().getHiddenNeurons()
        .getNeuronCountExcludingBias(); j++) {
      UndirectedLayerContext hiddenNeuronInspectionContext =
          new UndirectedLayerContextImpl(0, matrixFactory, false);
      NeuronsActivation neuronActivationMaximisingActivation =
          restrictedBoltzmannMachine.getLayer().getOptimalVisibleActivationsForHiddenNeuron(j,
              hiddenNeuronInspectionContext, matrixFactory);
      float[] neuronActivationMaximisingFeatures =
          neuronActivationMaximisingActivation.getActivations(matrixFactory).getRowByRowArray();

      float[] intensities = new float[neuronActivationMaximisingFeatures.length];
      for (int i = 0; i < intensities.length; i++) {
        double val = neuronActivationMaximisingFeatures[i];
        double boundary = 0.02;
        intensities[i] = val < -boundary ? 0f : val > boundary ? 1f : 0.5f;
      }
      MnistUtils.draw(intensities, display);
      Thread.sleep(100);
    }
    
    LOGGER.info("Generating sample data using Gibbs sampling..");
    Matrix visibleActivations = testDataInputActivations.getActivations(matrixFactory).getColumn(5);

    for (int k = 0; k < 10000; k++) {

      RestrictedBoltzmannSamplingActivation samplingResult =
          restrictedBoltzmannMachine.performGibbsSampling(
              new NeuronsActivationImpl(visibleActivations,
                  NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET),
              500, samplingContext);

      NeuronsActivation finalReconstruction = samplingResult
          .getLastVisibleNeuronsReconstructionLayerActivation().getVisibleActivationProbablities();

      visibleActivations = finalReconstruction.getActivations(matrixFactory);

      float[] reconstructionFeatures = finalReconstruction.getActivations(matrixFactory).getRowByRowArray();

      float[] intensities = new float[reconstructionFeatures.length];
      for (int i = 0; i < intensities.length; i++) {
        float val = reconstructionFeatures[i];
        intensities[i] = val;
      }
      MnistUtils.draw(intensities, display);
    }
  }
}
