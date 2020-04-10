# Feed-Forward-Neural-Net-Framework
 A general framework for traditional MLPs with many activations function/loss function choices provided out of the box.
 Completely written from scratch (Using Numpy etc.)

# Autodiff
Reverse-Mode Automatic Differentiator for Backpropagation. Create entry as nodes of a computational graph or parts of layers of a neural net, with provided functions. Provided functions include: softmax, dot (weighted sum with bias), hyperbolic tangent, sigmoid, addition, multiplication, cross-entropy loss calculation, regularization. Initialize the Wengert tape with all the nodes as a list and valMap with all initial values. WengertList.propagate() will forward propagate and calculate the output of the networks given the current state of the weights. WengerList.backpropagate() will backpropagate and calculate necessary gradients automatically.

# ANN
A simple 3-layer network with 20 for input layer, 100 neurons in hidden layer, and 26 in output for optical character classification. Translational, Rotational, and Scale invariant features extracted using the Radial Sector Coding algorithm. Trained by full gradient descent with adjustable learning rate. Provided weights.data and bias.data should get testing accuracy of around 80%. Weights initialized by Xavier-He initialization.

# RSC
Radial Sector Coding -- refer to https://www.jstage.jst.go.jp/article/softscis/2008/0/2008_0_1367/_article

Used the Chars74K dataset of fonts
