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

# mathematical mysteries of convolutional networks
Let's take a classification problem, we want to find a minimum of some invariants features to do classification efficiently. Indeed, reducing the dimensionality means that we will need less examples to learn how to do our classification.
In low dimension, it's quite easy to do some *linear interpolation*, but it's not possible in big dimensionality as the distance between two example is too big. 

Another basic idea is to do a *linear projection*, but most of the time, it will lead to loss of informations because of non-linearities. However, we can first do a transformation to linearize the problem and then do the projection. An example of this is **kernel classifiers** :

 1. Find a change of variable $\Phi(x) = \{ \phi_k(x) \}_{k \leq d^{prim}}$ it leads to linearization separation
 2. Find a linear projection $\langle \Phi(x),w \rangle = \sum_k w_k \phi_k(x)$

<figure>
  <div style="text-align: center">
    <img style="display: inline;" src="{{ site.baseurl }}/public/deep_learning_lecun_cdf/kernel_classifier.png" alt="kernel classifier">
          <figcaption> Fig1. kernel classifier </figcaption>
  </div>
</figure>

However, it is really difficult to find such a $\Phi$ ! This is where neural network comes: We make our $\Phi$ as a combination of convolutional nodes ( linear convolutions + non-linear scalar). Let's try to understand what are the mathematics behind it.
Why hierarchical network cascade? why convolutions? why indtroducing non-linearities? What are the properties of the learned linear operators $L_j$ ? Intuition of how it works is a we go deeper we kill variablility and create invariants! It works a bit like mathematical wavelet filter.

Conclusions we can take from Stephane Mallat presentation are :

 - Channel connections linearize other symmetries
 - Invariance to rotations are computed by convolutions
 - The convolution network operators $L_j$ have many roles (but difficult to separate these roles when analyzing learned network):
    - Linearize non-linear transformations (symmetries)
    - Reduce dimension with projections
    - Memory storage of « characteristic » structures
 - Deep convolutional networks have spectacular high-dimensional approximation capabilities ( notions of complexity, regularity, approximation theorems):
    - Seem to compute hierarchical invariants of complex symmetries
    - Used as models in physiological vision and audition
    - Close link with particle and statistical physics
    - Outstanding mathematical problem to understand them:



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

# Optimization or neural network

look for Markov model

better learn bias a bit faster than weights. scaling all biases by $\sqrt{n}$ seems to work well enough...

Training on data x is not equivalent to training on data 1 − x . A sigmoid learn faster when there is 1 instead of 0 on input. Using sigmoid or tanh is closely related to the x vs 1 − x problem.

### Can we avoid model rewriting?
Going back to the original recurrent NN model for text, there are dozens (actually: an infinite number) of such possible rewritings:
Two strategies:

- Try to identify variables that work best (eg: LSTMs)
 - Use a learning algorithm that is insensitive to model rewriting: an “invariant” algorithm

Classical such algorithm: Amari’s natural gradient using the inverse Fisher metric. But its algorithmic cost is O((#params) 2 dim(output)) per data point. => **Unusable for neural networks unless you have ≈ 100 neurons or #independent parameters ≪ network size.**


### Why invariant training algorithms?

Invariance often provides:

 - Fewer arbitrary choices, no model rewriting, fewer magic numbers
 - Often, better performance
 - **performance transfer**: good performance observed in a particular set of experiments automatically applies to a whole class of equivalent problems under a certain group of transformations


### Gradient descent: a physicist’s viewpoint
Gradient descent on loss function L(θ) iterates with $\theta^{k+1} = \theta^k - \eta \frac{\partial L(\theta^k)}{\partial \theta}$
If we do a dimensional analysis, unit of $\frac{\partial L}{\partial \theta} = \frac{unit\ of\ L}{unit\ of\ \theta}$ . So the gradient descent $\theta \leftarrow \theta - \eta \frac{\partial L}{\partial \theta}$ **is not homogeneous unless** $unit\ of\ \eta = unit\ of\ \theta^2$ 

Indeed, for small learning rate $\eta$, the gradient descent computation is equivalent to $\theta^{k+1} = arg \min_\theta \( L(\theta) + \frac{1}{d\eta} \Vert \theta - \theta^k \Vert^2 \)$ which depends on the numerical representation of $\theta$! Even a change of basis for $\theta$ is not neutral (ex scaling some components of $\theta$).

nvariant gradient descents: 
Solution: use a norm $\Vert \theta - \theta^{prim} \Vert$ that depends on what the network does, rather than how $\theta$ and $\theta^{prim}$ are represented as numbers for example **Riemannian metrics** are norms $\Vert \partial \theta \Vert^2 = \partial \Theta^T M(\theta) \partial \theta$ (for $\theta^{prim} = \theta + \partial \theta$ infinitesimally close to $\theta$) with $M(\theta)$ a well-chosen positive definite matrix depending on $\theta$ It leads to :

 - Riemannian gradient descent $\theta^{k+1} = \theta^k - \eta M(\theta)^{-1} \frac{\partial L(\theta^k)}{\partial \theta}$
 - For natural gradient $M(\theta)$ = Fisher information matrix

Two possible strategies to build a metric M(θ) and norm ‖θ − θ ′ ‖ M to obtain invariance wrt the representation of θ, in a scalable way:

 1. Try to *scale down the natural gradient* defined by the Fisher matrix. See also [Martens–Grosse 2015]. Related to AdaGrad without the square root. [Le Roux–Manzagol–Bengio 2007] use a small-rank reduction, but this *breaks invariance*.
 2. Try to *backpropagate a metric on the output*, layer by layer, until you get a metric on the parameters. Related to backpropagation with square weights used in the diagonal Gauss–Newton approximation of the Hessian [LeCun–Bottou–Orr–Müller 1996] , but keeping only the diagonal breaks invariance.

### Neural networks: fancy differential geometric setting

A Neural network is a  finite, directed acyclic graph L (“units”, “neurons”). Each unit $i \in L$ has an activity $a_i \in A_i$ where $A_i$ is some abstract manifold.

Activities for each input $x_k$ are computed by propagating $a_i = f_{i,\theta_i} ((a_j)_{j \rightarrow i})$

using an activation function $f_i : \Theta \times \prod\limits_{j \rightarrow i} A_j \rightarrow A_i$ depending on trainable parameters $\theta_i$ in a manifold $\Theta_i$

$\Rightarrow$ Any algorithm defined in this language has a performance that is independent from the manifold coordinates used to implement it
But the gradient descent is not well-defined in this space.


slide 73/115 of presentation 003-yann-ollivier







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


## Elastic Average SGD  (EASGD)
For distributing SGD over multiple CPU/GPU nodes

 - Expected Loss : $\min\limits_x F(x) := E[f(x,\xi)]$
 - Distributed form : $\min\limits_{x^1, \cdots, x^\rho} \sum\limits_{i=1}^r E[f(x^i,\xi^i)] + \frac{\rho}{2} \Vert x^i - \bar x \Vert^2$

with $\rho$ workers . Each worker get samples of the full dataset. They run momentum SGD asynchronously with $L^2$ penalty $\rho \Vert x^i - \bar x \Vert^2$ . Then worker i & master node 9server) communicates weights and server updates x slowly $x = x + \alpha(x^i -x)$.

$\frac{\rho}{2} \Vert x^i - \bar x \Vert^2$ is an elastic term which prevent workers from diverging to much from the central node.

This is parallelization by data! For more information, look for **Elastic Average Momentum SGD**.


# Target prop
The idea is propagate targets instead of gradients . It is used in RNN for speech recognition for example.
First, let's do a lagrangian formulation of backprop :

$L(Z,\lambda,W) = D(G_{S-1}, Z_S) + \sum\limits_{i=1}^S \lambda_i^T [ Z_i - G_i (Z_{i-1}, W_i)]$ with

 - $D(G_{S-1}, Z_S)$ cost function we want to minimize
 - $Z_S$  desired output
 - layers are $i$ indices,
 - $\sum\limits_{i=1}^S \lambda_i^T [ Z_i - G_i (Z_{i-1}, W_i)]$ connections in the network exprimed as constraints saying that output of layer $i$ should be equal to input of layer $i+1$ which is $Z_i$
  - $\lambda_i^T$ is a Lagrange multiplier (which is a vector)

So to do constraint optimization, we want to minimize the lagrangian relative to variables that interest us and we maximize is relative to Lagrange multiplier. So we compute the gradient

 - relative to the lagrange multiplier : $\partial(L(Z,\lambda, W)/ \partial lambda^k = 0 \ \Rightarrow \  Z_i = G_i(Z_{i-1}, W_i)$
 - relative to $Z_{i-1} :  $ $\partial(L(Z,\lambda, W)/ \partial Z_{i-1}^k = 0 \ \Rightarrow \  \lambda_{i-1} = [\partial G_i(Z_{i-1}, W_i)/\partial Z_{i-1}] \lambda_i$

We can then relax the constraint: Zi becomes a "virtual target" for layer i

$L(Z,W) = D(G_{S-1}, Z_S) + \sum\limits_{i=1}^S \alpha_i \Vert Z_i - G_i (Z_{i-1}, W_i) \Vert^2$

It now requires an optimization step with respect to virtual targets Z. It allows us to add penalties Z, such as sparsity, quantization..


#Other network

## Gating and attention
Connections are activated depending on context. Input of a unit is selected among several by the softmax output of s sub-network (the unit pay attention to a particular location)

Used for pictures legend,  speech translation

## Memory-augmented neural nets
Recurrent networks cannot remember things for very long (the cortex only remember things for 20 seconds), so we need a "hippocampus" (a separate memory module). Ex:

 - differentiable memory : stores Key-Value pairs (Ki, Vi) and take as input (address) X. Then, define coefficients $C_i = \frac{e^{K_i^T X}}{\sum\limits_j e^{K_j^T X}}$  and the output $Y = \sum\limits_i C_i V_i$
 - stack-augmented RNN [Joulin & Mikolov, ArXiv:1503.01007]

# Unsupervized Learning
The three missing pieces for AI (besides computation) are:

 1. Integration, representation/Deep Learning with reasoning, Attention, planning and memory ( a few bits for some samples = cherry)
 2. Integrating supervized, unsupervized and reinforcement learning into a single "algorithm" ( 10 to 10k bits per sample = icing)
 3. Effective ways to do unsupervized learning ( millions of bits per sample = cake)

Most of the learning performed by animals and humans is unsupervised, we build a model of the world through perdictive unsupervised learning. This predictive model gives us "common sense".

unsupervised learning can be based on reconstruction or prediction. Indeed, reconstruction is just prediction with delay 0 but the world is only partially predictable.
Even if the world were noiseless and quasi deterministic, our current methods would fail. Our failure to do unsupervised learning has nothing to do with our inability to model probability distributions in high-dim spaces, it has to do with our inability to capture complex regularities.


One of the difficulty in unsupervized learning is that the output is a set (or a distribution) as the training samples are mereley representatives of a whole set of possible outputs.
Question how do we design a loss function so that the machine is enticed to ouput points on the data manifold, but not punished for producing points different from the desired output in the training set?

One strategy is **energy-based unsupervised learning**. It consist of learning an *energy function* that takes low values on the data manifold and higher values everywhere else

We could also try to transform energies into probabilities :

 - energy can be interpreted as an unnormalized log density
 - using a Gibbs distribution (beta parameter being akin to an inverse temperature) : $P(Y \vert W) = \frac{e^{-\beta E(Y,W)}}{\int_y e^{-\beta E(Y,W)}}$ and $E(Y,W) \propto -\log P(Y \vert W)$
 - However, the denominator is often intractable!! So using probabilities restrict us to certain cost function! $\Rightarrow$ don\t compute probabilities unless you absolutely have to

Here is eight strategies to shape the energy function:

 1. build the machine so that the volume of low energy stuff is constant *PCA, K-means, GMM, square ICA*
<figure>
  <div style="text-align: center">
    <img style="display: inline;" src="{{ site.baseurl }}/public/deep_learning_lecun_cdf/constant_volume_of_low_energy.png" alt="constant_volume_of_low_energy">
          <figcaption> Fig3. constant volume of low energy </figcaption>
  </div>
</figure>

 2. push down of the energy of data points, push up everywhere else *Max likelihood (needs tractable partition function)*
 3. push down of the energy of data points, push up on chosen locations *contrastive divergence, Ratio Matching, Noise Contrastive Estimation,Minimum Probability Flow*
 4. minimize the gradient and maximize the curvature around data points *score matching*
 5. train a dynamical system so that the dynamics goes to the manifold *denoising auto-encoder*
 6. use a regularizer that limits the volume of space that has low energy *Sparse coding, sparse auto-encoder, PSD*
 7. if $E(Y) = \Vert Y - G(Y) \Vert^2$, make $G(Y)$ as "constant" as possible. *Contracting auto-encoder, saturating auto-encoder*
 8. Adversarial training: generator tries to fool real/synthetic classifier.

<figure>
  <div style="text-align: center">
    <img style="display: inline;" src="{{ site.baseurl }}/public/deep_learning_lecun_cdf/energy_functions_of_various_methods.png" alt="energy functions of various methods">
          <figcaption> Fig4. energy functions of various methods </figcaption>
  </div>
</figure>


next : 15/04 


# References

[videos](https://www.college-de-france.fr/site/yann-lecun/course-2015-2016.htm)