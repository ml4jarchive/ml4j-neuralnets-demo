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

package org.ml4j.nn.demo.base.supervised;

import org.ml4j.MatrixFactory;
import org.ml4j.nn.NeuralNetworkContext;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.supervised.SupervisedNeuralNetwork;

/**
 * Base class for test harness to train and showcase a SupervisedNeuralNetwork.
 * 
 * @author Michael Lavelle
 *
 * @param <N> The type of SupervisedNeuralNetwork we are showcasing
 * @param <C> The type of runtime NeuralNetworkContext for this
 *            SupervisedNeuralNetwork
 */
public abstract class SupervisedNeuralNetworkDemoBase<N extends SupervisedNeuralNetwork<C, N>, C extends NeuralNetworkContext> {

	/**
	 * Run the demo.
	 * 
	 * @throws Exception If any exceptions occur
	 */
	public void runDemo() throws Exception {

		// Create a matrix factory we can use to create our NeuronsActivations
		// and contexts
		MatrixFactory matrixFactory = createMatrixFactory();

		// Create the training data NeuronActivations
		NeuronsActivation trainingDataInputActivations = createTrainingDataNeuronActivations(matrixFactory);

		// Create the training labels NeuronActivations
		NeuronsActivation trainingLabelOutputActivations = createTrainingLabelNeuronActivations(matrixFactory);

		// Determine the feature counts and bias inclusion of the training data
		int inputFeatureCount = trainingDataInputActivations.getActivations(matrixFactory).getColumns();

		// Create the neural network for this feature count and bias
		N supervisedNeuralNetwork = createSupervisedNeuralNetwork(inputFeatureCount);

		// Create the training context
		C trainingContext = createTrainingContext(supervisedNeuralNetwork, matrixFactory);

		// Train the network
		supervisedNeuralNetwork.train(trainingDataInputActivations, trainingLabelOutputActivations, trainingContext);

		// Showcase the network on the test set
		showcaseTrainedNeuralNetworkOnTrainingSet(supervisedNeuralNetwork, trainingDataInputActivations,
				trainingLabelOutputActivations, matrixFactory);

		// Create the test set
		NeuronsActivation testSetInputActivations = createTestSetDataNeuronActivations(matrixFactory);

		// Create the test set
		NeuronsActivation testSetLabelActivations = createTestSetLabelNeuronActivations(matrixFactory);

		// Showcase the network on the test set
		showcaseTrainedNeuralNetworkOnTestSet(supervisedNeuralNetwork, testSetInputActivations, testSetLabelActivations,
				matrixFactory);
	}

	/**
	 * Allows implementations to define the MatrixFactory they wish to use.
	 * 
	 * @return The MatrixFactory used for this demo
	 */
	protected abstract MatrixFactory createMatrixFactory();

	/**
	 * Constructs an input NeuronsActivations from the training data.
	 * 
	 * @param matrixFactory The MatrixFactory used to create the NeuronsActivations
	 *                      Matrix
	 * @return The input NeuronsActivations from the training data
	 */
	protected abstract NeuronsActivation createTrainingDataNeuronActivations(MatrixFactory matrixFactory);

	/**
	 * Constructs an output NeuronsActivations from the training labels.
	 * 
	 * @param matrixFactory The MatrixFactory used to create the NeuronsActivations
	 *                      Matrix
	 * @return The output NeuronsActivations from the training labels
	 */
	protected abstract NeuronsActivation createTrainingLabelNeuronActivations(MatrixFactory matrixFactory);

	/**
	 * Constructs an input NeuronsActivations from the test data.
	 * 
	 * @param matrixFactory The MatrixFactory used to create the NeuronsActivations
	 *                      Matrix
	 * @return The input NeuronsActivations from the test data
	 */
	protected abstract NeuronsActivation createTestSetDataNeuronActivations(MatrixFactory matrixFactory);

	/**
	 * Constructs an output NeuronsActivation from the test labels.
	 * 
	 * @param matrixFactory The MatrixFactory used to create the NeuronsActivations
	 *                      Matrix
	 * @return The output NeuronsActivations from the test labels
	 */
	protected abstract NeuronsActivation createTestSetLabelNeuronActivations(MatrixFactory matrixFactory);

	/**
	 * Constructs the SupervisedNeuralNetwork we are demonstrating, given the input
	 * data's featureCount.
	 * 
	 * @param featureCount The number of features in the input data this
	 *                     SupervisedNeuralNetwork supports
	 * @return the UnsupervisedNeuralNetwork we are demonstrating
	 */
	protected abstract N createSupervisedNeuralNetwork(int featureCount);

	/**
	 * Creates the NeuralNetworkContext we use in order to train the
	 * SupervisedNeuralNetwork.
	 * 
	 * @param supervisedNeuralNetwork The SupervisedNeuralNetwork we are training
	 * @param matrixFactory           The MatrixFactory we are using for this demo
	 * @return the NeuralNetworkContext we use in order to train the
	 *         SupervisedNeuralNetwork
	 */
	protected abstract C createTrainingContext(N supervisedNeuralNetwork, MatrixFactory matrixFactory);

	/**
	 * Method to be implemented by subclasses to showcase the
	 * SupervisedNeuralNetwork.
	 * 
	 * @param supervisedNeuralNetwork  The UnsupervisedNeuralNetwork we are
	 *                                 showcasing
	 * @param testDataInputActivations The input NeuronsActivation instance
	 *                                 generated by the test data
	 * @param testSetLabelActivations  The labels NeuronsActivation instance
	 *                                 generted by the test set
	 * @param matrixFactory            The MatrixFactory we are using for this demo
	 * @throws Exception In the event of an exception
	 */
	protected abstract void showcaseTrainedNeuralNetworkOnTestSet(N supervisedNeuralNetwork,
			NeuronsActivation testDataInputActivations, NeuronsActivation testSetLabelActivations,
			MatrixFactory matrixFactory) throws Exception;

	/**
	 * Method to be implemented by subclasses to showcase the
	 * SupervisedNeuralNetwork.
	 * 
	 * @param supervisedNeuralNetwork  The UnsupervisedNeuralNetwork we are
	 *                                 showcasing
	 * @param testDataInputActivations The input NeuronsActivation instance
	 *                                 generated by the test data
	 * @param testSetLabelActivations  The labels NeuronsActivation instance
	 *                                 generted by the test set
	 * @param matrixFactory            The MatrixFactory we are using for this demo
	 * @throws Exception In the event of an exception
	 */
	protected abstract void showcaseTrainedNeuralNetworkOnTrainingSet(N supervisedNeuralNetwork,
			NeuronsActivation testDataInputActivations, NeuronsActivation testSetLabelActivations,
			MatrixFactory matrixFactory) throws Exception;
}
