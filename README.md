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
