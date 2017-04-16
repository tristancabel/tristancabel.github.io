---
layout: postkatex
title: "deep learning :  Y. LeCun at College de France part 2"
categories: machine_learning
tags: bigdata machine_learning deep_learning
---

Notes from Y. LeCun lessons at College de France on Deep Learning part 2.
<!--more-->


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


## Learning to Infer LISTA
Iterative algorithm that converges to optimal sparse code

```
Y -> [ W_e ] --> [ + ] --> [ sh() ] --> [ Z ] -->
                   |                      |
                   |---<---[ S ]<----------

```


 - $Z(t+1) = Shrinkage_{\lambda/L} [ Z(t) - \frac{1}{L} W_d^T(W_d Z(t) - Y)]$
 - $Z(t+1) = Shrinkage_{\lambda/L}  [ W_e^T Y + S Z(t)]$ with $W_e = \frac{1}{L} W_d^T$ and $S= I - \frac{1}{L} W_d^T W_d$
 - S laterail Inhibition

So think of the FISTA flow graph as a recurrent neural net where $W_e$ and $S$ are trainable parameters. Learn We and S matrices with "backprop-through-time"

## adversarial learning

TODO



next : 15/04 

