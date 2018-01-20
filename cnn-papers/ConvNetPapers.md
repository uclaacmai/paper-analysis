## AlexNet Paper (https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)



#### Introduction

- Until recently (i.e. in 2012) computer vision datasets were quite small, compared to ImageNet, which contains 15 million labeled images in 22k categories
- Object recognition is a hard problem, so we can't rely on data alone to solve it
- Need a specialized model different from traditional DNNs that incorporate some assumptions/prior knowledge, which is what CNNs do
- CNNs are highly configurable, meaning that the number of convolutional and fully connected layers, and number of hidden units in each layer can be controlled
  - Due to the assumptions it makes about images (hierarchical structure, spatial invariance), CNNs typically have fewer parameters than DNNs of the same size, so they are easier to train
- Need a fast implementation of 2d convolution and a lot of GPUs



#### The Dataset

- ImageNet dataset: 15 million images belonging to one of 22k categories
- Preprocessing applied:
  - downsampled to fixed resolution (256 x 256)
  - pixel values were centered



#### The Architecture

- Used the ReLu nonlinearity instead of sigmoid/tanh

  - Simply can't do deep learning with sigmoid, the problem of vanishing gradients is just too big. 
  - ReLu gradients are 0 or 1, which means that they don't dramatically scale down the incoming gradient, solving the vanishing gradient problem
  - Easier/faster implementation: max(0, x)
  - Encourage sparsity in the model
  - Issues: "dead" ReLu's -> ReLus that never pass any activations through

- Network trainign was split across 2 GPUs

  - neurons were split across GPUs, only communicate at certain layers
  - chose this pattern with cross validation

- Local Response Normalization

  - Goal is to improve generalization
  - A type of "lateral inhibition" -> ability for an excited neuron to subdue/normalize its neighbors
  - Basically when a neuron has a very large activation, we want it to become even more sensitive/activate more when we use that input
    - so subdue the other neurons so that this one is relatively more excited
  - dampen responses of surrounding neurons
  - basically enhance the "peaks" and dampen the "flats" to model biological neurons and make neurons more sensitive to certain inputs

- Used overlapping pooling

- Overall architecutre

  - 8 layers: 5 conv and 3 fully connected, and then a 1000 way softmax
  - cross-entropy loss function, basically maximum likelihood approach

- Reducing overfitting:

  - One way: data augmentation/label preserving transformations: flip/rotate the image, extract random patches
  - Dropout: randomly set some proportion of the activations to 0
    - Means that the network cannot rely on specific presence of features/activations -> less overfitting
    - neurons learn more reduntant, robust representations
    - Each time a different architecture (that shares weights) is sampled, so it's like training an ensemble of correlated networks

  #### Details of Learning

  - network was trained with stochastic gradient descent using momentum
    - weight decay was also used (at each step $w_i$ was decayed by $w_i -= 0.0005*w_i$)
    - important to note that this was not just a regularizer, it actually decreased model training error
  - random initialization from a 0 mean Gaussian



## GoogLeNet Paper

- Note: The L is capitalized to pay homage to Yann LeCun (Facebook) who is the father of CNNs

  #### Motivation & High-Level Considerations

  - Easiest way to increase perf of DNNs is to make them longer and wider

    - more hidden layers, more hidden neurons

  - Comes at the expense of overfitting and vastly increased computational needs though

  - There's actually a quadratic increase in computation if the number of filters in 2 conv layers increases

  - Sparsely connected architectures is a good solutions

    - mimicks bio systems
    - Arora et al established theoretical work on modelling probability distributions of a dataset with a deep neural network

  - Conflict in use of sparsity that is supported by theoretical results, but our current hardware is optimized for operations on dense collections:

    - very fast 2d convolutions and dense matmults

      ​

    #### Architecture Details

    - Overarching theme of Inception: create a network architecture that approximates a sparse structure, by only using dense components so we can take advantage of current hardware
    - Theoretical goal: build a network layer by layer
      - look at the correlation statistics of the last layer, and combine the highly correlated units (neurons)
      - These groups become the units in the next layer 
    - Sparsely connected neurons in a deep network will represent the probability distribution of a dataset, observe correlations in activations to construct network layer by layer
    - Main idea of inception: at each conv layer, do a bunch of different ops:
      - 1x1 conv, 3x3 conv, 5x5 conv, max pool and then concatenate the output of all of them
    - Naive approach of doing straight 3x3 convs and 5x5 cones results in parameter blowups and computational inefficienty
    - Want to keep representations from blowing up, so 1 x 1 convolutions were used

    #### GoogLe Net

    - Use of 1x1 convolutions was really important for dimensionality reduction
    - Network was 22 layers deep, which introduced the concern that backpropagation may lead to vanishing gradients/slower learning
      - Used auxiliary classifiers in the beginning/middle/end of the network to improve the gradient signal
      - The gradients from these classifiers were added to the true gradient, but discounted by a weight

## ResNet Paper

- 152 layers deep (wow!)
- In traditional learning, we have an input x and then we do some series of operations (typically conv->relu->pooling) to get the output F(x).
- With residual learning, we basically want to remember the input into the previous layer and use that
- So we create a new $H(x)$ and let it equal $F(x)$ (the output from our layer) $+ x$ (the input into the previous layer)
- The authors hypothesized that it is easier to optimize residual mappings than the original mappings

#### Deep Residual Learning

- Degradation problem in traditional networks: 
  - If we keep stacking layers that must do identity mappings (i.e. the layer just outputs the input) then intuitively we wouldn't expect the error to get any worse
  - But according to a few experiments, it did, indicating that optimizers have trouble approximating identity mappings more than a few layers deep
  - Building block is y = F(x, W) + x
  - F(x, W) is the residual mapping to be learned, and it is equal to H(x) - x, and H(x) is what we want to output
- Implementation:
  - Repeated conv layers followed by occasional residual blocks, where previous input is added to that layer's output
  - Learning approach:
    - Used batch norm
    - SGD with batches of size 256, no dropout
    - Momentum & weight decay
    - Same 10-crop testing approach as AlexNet (extract 10 random 224 x 224 patches from the 256 x 256 image, and return the most likely label across those 10)