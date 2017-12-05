# ml4j-neuralnets-demo

## Creating a supervised NeuralNetwork

```

 // Create a simple 2-layer SupervisedFeedForwardNeuralNetwork
    
    // First, construct a MatrixFactory implementation.
    MatrixFactory matrixFactory = new JBlasMatrixFactory();
    
    // First Layer, takes a 28 * 28 grey-scale image as input (plus bias) - with every input Neuron connected to all 100 hidden Neurons 
    // This Layer applies a sigmoid non-linearity and does not use batch-norm

    FeedForwardLayer<?, ?> firstLayer = new FullyConnectedFeedForwardLayerImpl(
        new Neurons3D(28, 28 ,1, true), new Neurons(100, false), 
        new SigmoidActivationFunction(), matrixFactory, false);
    
    // Second Layer, takes the activations of the 100 hidden Neurons as input and produces activations of the 10 softmax output Neurons.

    FeedForwardLayer<?, ?> secondLayer = 
        new FullyConnectedFeedForwardLayerImpl(new Neurons(100, true), 
        new Neurons(10, false), new SoftmaxActivationFunction(), matrixFactory, false);

    // Neural Network
    SupervisedFeedForwardNeuralNetwork neuralNetwork 
          =  new SupervisedFeedForwardNeuralNetworkImpl(firstLayer, secondLayer);

```

## Training the NeuralNetwork

```

    double[][] trainingData = ... ; // Our training data - one row per training example.
    double[][] trainingLabels = ... ; // Our training label activations - one row per training example.
    
    // Create NeuronsActivation instances for the inputs and desired outputs.
    NeuronsActivation trainingDataActivations = new NeuronsActivation(matrixFactory.createMatrix(trainingData), NeuronsActivationFeatureOrientation.COLUMNS_SPAN_FEATURE_SET); 

    NeuronsActivation desiredOutputActivations = new NeuronsActivation(matrixFactory.createMatrix(trainingLabels), NeuronsActivationFeatureOrientation.COLUMNS_SPAN_FEATURE_SET); 
    
    // Create a context to train the network from Layer 0 to the final Layer.
    FeedForwardNeuralNetworkContext context = 
        new FeedForwardNeuralNetworkContextImpl(matrixFactory, 0, null);
    
    // Configure the context to train in mini-batches of 32 and to run for 100 Epochs.
    context.setTrainingMiniBatchSize(32);
    context.setTrainingEpochs(100);
    
    // Train the NeuralNetwork
    neuralNetwork.train(trainingDataActivations, desiredOutputActivations, context);

```

## Using the NeuralNetwork

```

    // Use the NeuralNetwork, to obtain output activations for test set data activations.
    
    double[][] testSetData = ...  ; // Our test set data - one row per training example.

    // Create NeuronsActivation instance from this test set data.
    NeuronsActivation testSetDataActivations = new NeuronsActivation(matrixFactory.createMatrix(trainingData), NeuronsActivationFeatureOrientation.COLUMNS_SPAN_FEATURE_SET); 
    
    // Obtain the output NeuronsActivation by forward propagating the input activations through the Network
    NeuronsActivation outputActivations = 
          neuralNetwork.forwardPropagate(testSetDataActivations, context).getOutputs();
    
    // Use the output activations (eg. to classify, by taking the argmax of each row)

```
