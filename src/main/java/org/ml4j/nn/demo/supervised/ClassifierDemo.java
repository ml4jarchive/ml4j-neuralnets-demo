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
import org.ml4j.nn.demo.util.MnistUtils;
import org.ml4j.nn.demo.util.PixelFeaturesMatrixCsvDataExtractor;
import org.ml4j.nn.demo.util.SingleDigitLabelsMatrixCsvDataExtractor;
import org.ml4j.nn.layers.DirectedLayerContext;
import org.ml4j.nn.layers.FeedForwardLayer;
import org.ml4j.nn.layers.FullyConnectedFeedForwardLayerImpl;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.Neurons3D;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.neurons.NeuronsActivationFeatureOrientation;
import org.ml4j.nn.supervised.FeedForwardNeuralNetworkContextImpl;
import org.ml4j.nn.supervised.SupervisedFeedForwardNeuralNetwork;
import org.ml4j.nn.supervised.SupervisedFeedForwardNeuralNetworkImpl;
import org.ml4j.util.DoubleArrayMatrixLoader;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * A simple test harness to train and showcase a Classifier.
 * 
 * @author Michael Lavelle
 */
public class ClassifierDemo
    extends SupervisedNeuralNetworkDemoBase<SupervisedFeedForwardNeuralNetwork, 
    FeedForwardNeuralNetworkContext> {

  private static final Logger LOGGER = LoggerFactory.getLogger(ClassifierDemo.class);

  public static void main(String[] args) throws Exception {
    ClassifierDemo demo = new ClassifierDemo();
    demo.runDemo();
  }

  @Override
  protected SupervisedFeedForwardNeuralNetwork 
      createSupervisedNeuralNetwork(int featureCount) {

    // Construct a 2 layer Neural Network
    
    MatrixFactory matrixFactory = createMatrixFactory();
    
    FeedForwardLayer<?, ?> firstLayer = new FullyConnectedFeedForwardLayerImpl(
        new Neurons3D(28, 28 ,1, true), new Neurons3D(20, 20, 1, false), 
        new SigmoidActivationFunction(), matrixFactory, false);
    
    FeedForwardLayer<?, ?> secondLayer = 
        new FullyConnectedFeedForwardLayerImpl(new Neurons3D(20, 20, 1, true), 
        new Neurons(10, false), new SoftmaxActivationFunction(), matrixFactory, false);

    return new SupervisedFeedForwardNeuralNetworkImpl(firstLayer, secondLayer);
  }

  @Override
  protected NeuronsActivation createTrainingDataNeuronActivations(MatrixFactory matrixFactory) {
    LOGGER.trace("Creating training data NeuronsActivation");
    
    DoubleArrayMatrixLoader loader = new DoubleArrayMatrixLoader(
            ClassifierDemo.class.getClassLoader());
    // Load Mnist data into double[][] matrices
    double[][] trainingDataMatrix = loader.loadDoubleMatrixFromCsv("mnist2500_X_custom.csv",
            new PixelFeaturesMatrixCsvDataExtractor(), 0, 1000);
    
    return new NeuronsActivation(matrixFactory.createMatrix(trainingDataMatrix),
        NeuronsActivationFeatureOrientation.COLUMNS_SPAN_FEATURE_SET);
  }

  @Override
  protected NeuronsActivation createTestSetDataNeuronActivations(MatrixFactory matrixFactory) {
    LOGGER.trace("Creating test data NeuronsActivation");
    
    DoubleArrayMatrixLoader loader = new DoubleArrayMatrixLoader(
        ClassifierDemo.class.getClassLoader());
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
  protected FeedForwardNeuralNetworkContext createTrainingContext(
      SupervisedFeedForwardNeuralNetwork supervisedNeuralNetwork, MatrixFactory matrixFactory) {
    LOGGER.trace("Creating FeedForwardNeuralNetworkContext");
    // Train from layer index 0 to the end layer
    FeedForwardNeuralNetworkContext context =
        new FeedForwardNeuralNetworkContextImpl(matrixFactory, 0, null);
    context.setTrainingEpochs(200);
    context.setTrainingLearningRate(0.05);
    context.getLayerContext(0).getSynapsesContext(0).getAxonsContext(0, 0)
    .setLeftHandInputDropoutKeepProbability(0.8);
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
      MnistUtils.draw(intensities, display);
      Thread.sleep(100);
    }
    
    // Visualise the reconstructions of the input data
  
    LOGGER.info("Visualising data and classifying");
    for (int i = 0; i < 100; i++) {

      // For each element in our test set, obtain the compressed encoded features
      Matrix activations = testDataInputActivations.getActivations().getRow(i);
      
      NeuronsActivation orignalActivation = new NeuronsActivation(activations,
          NeuronsActivationFeatureOrientation.COLUMNS_SPAN_FEATURE_SET);

      MnistUtils.draw(orignalActivation.getActivations().toArray(), display);

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
        ClassifierDemo.class.getClassLoader());
    // Load Mnist data into double[][] matrices
    double[][] testDataMatrix = loader.loadDoubleMatrixFromCsv("mnist2500_labels_custom.csv",
        new SingleDigitLabelsMatrixCsvDataExtractor(), 0, 1000);
   
    return new NeuronsActivation(matrixFactory.createMatrix(testDataMatrix),
        NeuronsActivationFeatureOrientation.COLUMNS_SPAN_FEATURE_SET);
  }

  @Override
  protected NeuronsActivation createTestSetLabelNeuronActivations(MatrixFactory matrixFactory) {
    DoubleArrayMatrixLoader loader = new DoubleArrayMatrixLoader(
        ClassifierDemo.class.getClassLoader());
    // Load Mnist data into double[][] matrices
    double[][] testDataMatrix = loader.loadDoubleMatrixFromCsv("mnist2500_labels_custom.csv",
        new SingleDigitLabelsMatrixCsvDataExtractor(), 1000, 2000);
   
    return new NeuronsActivation(matrixFactory.createMatrix(testDataMatrix),
        NeuronsActivationFeatureOrientation.COLUMNS_SPAN_FEATURE_SET);
  }
}
