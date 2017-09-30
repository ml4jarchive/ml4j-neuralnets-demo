/*
 * Copyright 2017 the original author or authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.ml4j.nn.demo.unsupervised;

import org.ml4j.MatrixFactory;
import org.ml4j.layers.mocks.FeedForwardLayerMock;
import org.ml4j.mocks.MatrixFactoryMock;
import org.ml4j.nn.demo.base.unsupervised.UnsupervisedNeuralNetworkDemoBase;
import org.ml4j.nn.layers.FeedForwardLayer;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.neurons.NeuronsActivationFeatureOrientation;
import org.ml4j.nn.unsupervised.AutoEncoder;
import org.ml4j.nn.unsupervised.AutoEncoderContext;
import org.ml4j.nn.unsupervised.mocks.AutoEncoderContextMock;
import org.ml4j.nn.unsupervised.mocks.AutoEncoderMock;
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
    
    FeedForwardLayer<?> encodingLayer = new FeedForwardLayerMock();
    FeedForwardLayer<?> decodingLayer = new FeedForwardLayerMock();

    return new AutoEncoderMock(encodingLayer, decodingLayer);
  }

  @Override
  protected NeuronsActivation createTrainingDataNeuronActivations(MatrixFactory matrixFactory) {
    LOGGER.trace("Creating training data NeuronsActivation");
    // Dummy data for now
    return new NeuronsActivation(matrixFactory.createZeros(1000, 784), false,
        NeuronsActivationFeatureOrientation.COLUMNS_SPAN_FEATURE_SET);
  }

  @Override
  protected NeuronsActivation createTestSetDataNeuronActivations(MatrixFactory matrixFactory) {
    LOGGER.trace("Creating test data NeuronsActivation");
    // Dummy data for now
    return new NeuronsActivation(matrixFactory.createZeros(1000, 784), false,
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
  protected void showcaseTrainedNeuralNetwork(AutoEncoder unsupervisedNeuralNetwork,
      NeuronsActivation testDataInputActivations, MatrixFactory matrixFactory) throws Exception {
    LOGGER.info("Showcasing trained AutoEncoder...");
  }
}
