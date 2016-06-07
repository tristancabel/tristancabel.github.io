---
layout: postkatex
title: "deep learning :  Y. LeCun at College de France"
categories: machine_learning
tags: bigdata machine_learning deep_learning
---

Notes from Y. LeCun lessons at College de France on Deep Learning.
<!--more-->

# Introduction

Trainable Feature hierarchy
The hierarchy of representations changes while increasing the level of abstraction. Each stage is a kinf of trainable feature transform. For example:

 - image recognition *pixel -> edge -> texton -> motif -> part -> object*
 - text *character -> word -> word group -> clause -> sentence -> story*
 - speech


Deep learning addresses the problem of learning hierarchical representations with a single algorithm (or perhaps a few algorithms)



There is three types of deep architectures

 - **Feed-forward** "->" : calculate the output from the input like multilayer neural nets, convolutional nets   
 - **Feed-Back** "<-"  : find an output that fit the input like stacked sparse coding, deconvolutional Nets 
 - **Bi-Directional** "<->" : Deep Boltzmann Machines, Stacked Auto-Encoders

Then, there is three types of training protocols:

 1. **Purely supervized** typically trained with SGD, using backprop to compute gradients. Used in most practical systems for speech and image recognition
 2. **Unsupervised, layerwise + supervized classifier on top** which train each layer unsupervized, one after the other and train a supervized classifier on top, keeping the other layers fixed. Good when very few labeled samples are available.
 3. **Unsupervised, layerwise + global supervised fine-tuning** which train each layer unsupervized, one after the other and add a classifier layer, and retrain the whole thing supervised. Good when label set is pood (e.g. pedestrian detection)

Unsupervised pre-training often uses regularized auto-encoders.

## Do we really need deep architectures?
*We can approximate any function as close as we want with shallow architecture. Why would we need deep ones?*
 kernel machines: $y = \sum\limits_{i=1}^P \alpha_i K(X, X^i)$  or with 2-layer neural net: $y=F(W^1 . F(W^0 . X))$

The answer is, deep machines are more efficient for representing certain classes of functions, particularly those involved in visual recognition. They can represent more complex functions with less "hardware".

For example, let's take a $K$ comparing the ressemblance between two pictures; let's say a picture of a chair. The problem is, if we rotate the chair by 1 degree, the comparison pixel by pixel won't work. same if we change the luminosity, or the type of chair. It means that we will need a gigantic P to handle all these cases!

Another example, let's take a N-bit parity classifier, output = 1 if the number of bits is pair, 0 otherwise. If we want to solve this using only 2 layers, it will requires an exponential number of gates. However, it only requires N-1 XOR in a tree of depth log(N).

## Which models are deep
2-layer models as Neural nets with one hiddend layer  are not deep because there is no feature hierarchy. Indeed, it can be expressed as $G(X,\alpha) = \sum\limits_j \alpha_j K(X^j, X)$

Deep learning is non-convex: the order of layers is important!

## How to choose layers
We have to find independant explicative parameters.

Let's take all face images of a person, each picture being  1000x1000 pixels = 1,000,000 dimensions. But the face has 3 cartesian coordinates and 3 Euler angles, and gumans hace less than about 50 muscles in the face. Hence, the manifold of face images for a person has <56 dimensions!  But **we do not have good and general methods to learn functions that turns an image into this kind of representation.**

But we can do dome invariant feature learning: embed the input non-linearly into a higher dimensional space because in this new space, things that were non separable may become separable. Then pool regions of the new space together (bring together things that are semantically similar or aggregate over space or feature type).

` 1_Layer = { Input -> [Non-Linear Function] -> high-dim features -> [Pooling] -> stable/invariant features }`

The most used pooling methods are:

 - max $max_i(X_i)$
 - Lp norm $L_p : \sqrt[p] X_i^p$
 - log probability $PROB: \frac{1}{b} \log \big( \sum\limits_i e^{h X_i} \big)$


## Back propagation
Let's take an input $X$ and a desired output $Y$. We will use a simple sequential/layered feed-forward architecture. At each layer $i$ , there will be a function $F_i(X_{i-1}, W_i)$ applying a transformation from the given input with weights $W_i$.

Then, the last stage is a classifier $C(X_n, Y)$, and the all can be described by a cost function $E(W,Y,X)$

To train a multi-module system, we must compute the gradient of $E(W,Y,X)$ with respect to all the parameters in the system ( the $W_k$) : the $\frac{\partial E}{\partial X_k}$ .

Then we can apply chain rule to compute:

 - $\frac{\partial E}{\partial W_k} = \frac{\partial E}{\partial X_k} \frac{\partial F_k(X_{k-1}, W_k)}{\partial W_k}$ with dimensions  $[1 \times N_w] = [1 \times N_x] . [N_x \times N_w]$
 - $\frac{\partial E}{\partial X_{k-1}} = \frac{\partial E}{\partial X_k} \frac{\partial F_k(X_{k-1}, W_k)}{\partial X_{k-1}}$

$\frac{\partial F_k(X_{k-1}, W_k)}{\partial W_k}$ being the *Jacobian matrix of* $F_k$ with respect to $W_k$ and $\frac{\partial F_k(X_{k-1}, W_k)}{\partial X_{k-1}}$ being the *Jacobian matrix of* $F_k$ with respect to $X_{k-1}$. $F_k$ has two jacobian matrices because it has two arguments.

The back propagation consists of recurring these equations to compute all the $\frac{\partial E}{\partial W_k}$ for $k \in [1,n]$.
One optimization is to use stochastic gradient descent on minibatches instead of sequential values to be able to do parallelization. It's also good to use "dropout" for regularization during training (consist of randomly saying that some of units don't exist and compute the output to ensure that the system is not dependant on one particular unit).

## Typical multilayer Neural Net Architecture
Complex learning machines can be built by assembling **modules** into networks. Some examples of modules can be:

 - linear module $Out = W.In+B$
 - ReLU Module (Rectified Linear Unit) $Out = 0 if in<0 ; Out = In \space otherwise$ . Nowadays, ReLU are more used than sigmoid because of the small gradient of sigmoid at some points leading to a small learning rate. 
 - Cost Module: squared distance $C= \|\|In_1 - In_2 \|\|^2$
 - cross entropy (good for classification)

# Convolutional net (CNN)

As an **introduction**, let's take a module taking a time-delayed input $R = F(X_t, X_{t-1}, X_{t-2}, W)$ and we want to have $Y_{t+1} = R$. For example, predict the next character or word in a text, or predict the evolution of stock-market.  We can see that the unit computing $R = F(X_t, X_{t-1}, X_{t-2}, W)$ at instant $t$ is the same that will be used as $t+1$ so weights will be the same. The idea of replicate units is quite used in visual recognition to detect a motif in different parts of a picture (detect a digit in check recognition for example) using **convolution**.

These past few years, the deepness of convolutional nets have increased a lot (ResNet 152 levels, with identity links between even levels to allow  efficient back-propagation). This allow the system to act as an iteratif system.

The basic idea to recognize a form into an image is to apply the same network with a sliding window, however, this implies a window of fized size. Nowadays, there is more efficient modern methods: for example, another idea is to add 4 outpus to the network defining the position of the object inside the window! Another idea is RCNN, 1 network do a detection, then another network look specifically inside this first window.

A new method for localizing objects is **DeepMask** by *Pinheiro, Collobert, Dollar ICCV 2015* . The idea is not to make a box like earlier but to make a mask of the object. So the system, will be simultanously trained to make an object recoginition and a mask of that object.


Convolutional nets are specially good for:

 - signals that comes in the form of multidimensional arrays
 - signals that have a strong local correlation
 - signals where features can appear anywhere
 - signals in which objects are invariant to translations and distortions

Some 1D ConvNets examples are Text classification, musical genre prediction; some 2D examples are object detection in images, recognition in audio; some 3D examples are video recognition, biomedical volumetric images.
Convolutional nets can also be used to perform monuments recognition, face reconstruction and identification, driver assistance ..



# Recurrent Network (RNN)

<table >
  <col width="50%">
  <col width="50%">
  <tbody>
    <tr>
      <td style="background:white;"> <ul>
        <li>For recurrent network, the state at time $t$ depends on the state at time $t-1$. </li>
        <li> We can also do loop unrolling to turns recurrent net into feed-forward net with shared weights. </li> 
        <li> RNN can be made with "depth", to learn high-level representations of sequences. </li>
        <li> RNN are prone to vanishing gradient </li>
        <li> RNN are used to keep information about past events (like speech modelisation) and a false hypothetis was that they need memory to remenber these past information which is false: we need to make the network so that the computation made to calculate the current state from previous state is reversible (meaning we can compute $X(t)$ from $X(t+1$) = bijective function ). </li>
        <li> We can do some temporal subsampling to increase length of "memory" use a $Z(t-1)$ not to update $Z(t)$ but $Z(t+1)$ </li>
     
     </ul></td>
      <td style="background:white;">
      <figure>
        <div style="text-align: center">
          <img style="display: inline;" src="{{ site.baseurl }}/public/deep_learning_lecun_cdf/recurrent_network.png" alt="recurrent network">
          <figcaption> Fig1. recurrent network </figcaption>
        </div>
      </figure>
      </td>
    </tr>
  </tbody>  
</table>

A good to know RNN type is **LSTM (Long Short Term Memory)** to fix the long term dependency. The basic idea is that each unit has an input tie to an internal state which is then used to compute the output. And each internal state is connected to itself with a slighty less than 1 weight by a loop than can be active or not. So the unit can choose to "forget" or to "remember" the past.

With this type of network *Sutskever et al, NIPS 2014* made a translation language software. Language modeling is made using a combination of CNN and LSTM in *"Exploring the limits of Language Modeling" R. Jozefowicz, O. Vinyals et al*

# Energy-based Learning

<div class="row">
  <div class="col-sm-6 col-md-6 col-xs-6">
    <img class="pull-left" src="{{ site.baseurl }}/public/deep_learning_lecun_cdf/energy_based_models.png" alt="recurrent network">
  </div>
  <div class="col-sm-6 col-md-6 col-xs-6">
    <p>The basic idea is to do complex inference, for example when we want to do speech recognition, or translation where systems output a structured answer. There isn't a simple and some time there is different outputs possible (translation).
This model make a scalar value "energy" $E(Y,X)$ , measuring the compatibility between an observed variable X and a variable to be predicted Y. </p>
<p> What we want to know (the inference) is the $Y$ that minimizes the nergy within a set: $Y^* = argmin_{Y \in y} E(Y,X)$  similitude of *f(Input)* and *output*. </p>

<p>However, **energies are uncalibrated**, so energies of two separately-trained systems cannot be combined. To do so, we can turn them into probabilities using for example *Gibbs distribution* :  $P(Y\|X) = \frac{e^{-\beta E(Y,X)}}{ \int_{y \in Y} e^{-\beta E(y,X)}}$  with $\beta$ being the inverse temperature  and $\int_{y \in Y}$ the partition function.    The minus mean that low energy equal high probability. </p>

<p> Often, there is also a third variable $Z$ an "unobserved variable".For example, in face recognition in image with a sliding window, Z can be the position in the picture  </p>
  </div>
</div>

## training an EBM

Training an energy-based model, consists in shaping the energy function so that the energie of the correct answer is lower than the energies of all other answers (lowering the energie of the correct answer, and raising the energies of the other answers) : $E(animal,X) < E(y,X) \forall y \neq animal$

For this, the cost function  we want to minimize is $\Psi (E,S) = \frac{1}{P} \sum\limits_{i=1}^P L(Y^i, E(W,y,X^i)) + R(W)$ with:

 - $L()$ per-sample loss
 - $Y^i$ desired answer
 - $E(W,y,X^i)$ energy surface for a given $X^i$ as $Y$ varies
 - $R(W)$ Regularizer
 
There is four main steps to train an EBM:

 1. Design an architecture (a particular form of E(Z,Y,X)
 2. Pick an inference algorithm for Y: MAP or conditional distribution, belief prop, gradient descent, ..
 3. Pick a loss function in such a way that minimizing it with respect to W over a training set will make the inference algorithm fond the correct Y for a given X
 4. Pick an optimization method

The big problem is *what loss funciton will make the machine approach the desired behavior*

 - the **negative log-likelihood loss function** or Maximum mutual information is $L_{nll}(W,S) =  \frac{1}{P} \sum\limits_{i=1}^P ( E(Z, Y^i,X^i) + \frac{1}{\beta} \log \int\limits_{y \in Y} e^{-\beta E(Z, y, X^i)} )$  is reduces to the perceptron loss when Beta -> infinity
 - the **square-square loss function** is $L_{sq-sq} (Z, Y^i, X^i) = E(Z, Y^i, X^i)^2 + (max(0, m - E(Z, \bar Y^i, X^i)))^2$ , $m$ being a margin and $ \bar Y^i$ being chosen such as it's different from correct $Y^i$ but with the minimum energy ( aka the more threatening)
  
There is also *graph transformer network*

next : 25/03 


# References

[videos](https://www.college-de-france.fr/site/yann-lecun/course-2015-2016.htm)