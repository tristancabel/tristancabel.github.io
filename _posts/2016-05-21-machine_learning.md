---
layout: postkatex
title: "machine learning"
categories: machine_learning
tags: bigdata machine_learning statistics
---

This is the third lesson on Coursera big data specialization notes.
<!--more-->

There is four categories of machine learning:

 1. **supervized learning** feedback supplied explicitly
 2. **reinforcement learning** feedback supplied by the environment (ex: control theory, decision theory)
 3. **game theory** feedback supplied by other actors in the system
 4. **unsupervised learning** no feedback supplied

Learning is build across three core components: *Representation*, *Evaluation*, and *Optimization*.

 - *Representation* : what is your classifier? decision tree, neural network, hyperplane
 - *Evaluation* : how do we know if a given classifier is good or bad? precison and recall, squared error, likelihood
 - *Optimization* : how do you search among all the alternatives? greedy search? gradient descent?

## Supervised Learning
Supervized learning is when we want to learn the relationships between inputs and outputs given examples of inputs and associated outputs. We also use a different terminology depending on the attribute to learn:

 - **classification** if the attribute is categorical ("nominal")
 - **regression** if the attribute is numeric


For example, let's take the *titanic dataset*
we can for example take a representation such as : *IF sex='female' THEN survive=yes ELSE survive=no*
It gives us a confusion matrix (square array where diagonal is right)

<table>
  <thead>
    <tr>
      <td></td>
      <td>Predict No</td>
      <td>Predict Yes</td>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>True (No)</td>
      <td>468</td>
      <td>109</td>
    </tr>
    <tr>
      <td>True(Yes)</td>
      <td>81</td>
      <td>223</td>
    </tr>
  </tbody>
</table>

it gives us the **accuracy** $\frac{468+233}{468+233 + 81 + 109} = 79\%$ correct 

note that accuracy isn't always enough. You can't interpret *90% accuracy*, it depends of the problem. We need a baseline to compare against. For the future, note that *base rate* is accuracy of trivially predicting the most-frequent class; and *naive rate* is accuracy of some simple default or pre-existing model.

Let's take the following confusion matrix
<table>
  <thead>
    <tr>
      <td></td>
      <td>Predicted +</td>
      <td>Predicted -</td>
  </tr>
  </thead>
  <tbody>
    <tr>
      <td>True +</td>
      <td>a</td>
      <td>b</td>
    </tr>
    <tr>
      <td>True -</td>
      <td>c</td>
      <td>d</td>
    </tr>
  </tbody>
</table>

Some usefull terms are:

 - lift = $\frac{a/(a+b)}{(a+c)(a+b+c+d)}$
 - precision = $\frac{a}{a+c}$
 - recall = sensitivity = $\frac{a}{a+b}$
 - 1-specificity = $\frac{1-d}{c+d}$


### Entropy
In order to decide wheter a node in the decision tree is helpful or not we need the notion of **entropy** $H(X) = E(I(X)) = \sum_x p_x*I(x)$

For example, consider two sequences of coin flips THHTHTTHTH  and TTHTHTHHTH . How much *information* do we get after flipping each coin once?
We want some function "information" that satisfies: $information_{1and2}(p_1 p_2) = information_1(p_1) + information_2(p_2)$ so  it's  $I(X) = -\log_2 p_x$ 
The expected information is entropy: $H(X) = E(I(X)) = \sum_x{p_x*I(x)} = -\sum_x{p_x * log_2{p_x}}$
so for flipping a coin, entropy = -(0.5*log(0.5) + 0.5*log(0.5)) = 1 . So we learned 'one' bit of information

Now, let's consider rolling a die, $p_1 = \frac{1}{6} = p_2 = p_3 = ..$
Entropy = $-6*(\frac{1}{6} \log_2 \frac{1}{6}) ~ 2.58$ so after rolling a die, I gain more information than flipping a coin. It's more unpredictable.
Now what happens is we have a weighted die $p_1 = p_2 = p_3 = p_4 = p_5 = 0.1 p_6= 0.5$ , Entropy = $-5*(0.1\log_2 0.1) - (0.5\log_2 0.5) = 2.16$
So here we have less information after each try. It's less unpredictable

### gini coefficient
emtropy captured an intuition for "impurity". We want to choose attributes that split recored into pure classes. 
The **gini** coefficient measures inequality
$Gini(T) = 1 - \sum\limits_{i=1}^n p_i^2$

### decision tree
Back at the titanic example, we can go further making a *decision tree* such as:

```
IF pclass='1' THEN
    IF sex='female' THEN survive=yes
    IF sex='male AND age < 5 THEN survive=yes >
 ...
 
``` 

We usually choose the attribute with the highest information gain first (the bigger difference in entropy with our previous choice, so the on with the lower entropy) . 

Another parameter we nedd to pay attention is the number of levels of our decision tree: we don't want to have overfitting (perfectly fit train data but not test data) . Decision tree is prone to overfitting.

### overfitting
overfitting is when your learner outputs a classifier that is more accurate on the training data than on test data. Underfittiong often has high bias and low variance whereas overfitting has low bias and high variance.
When you have a some data to model your problem, split them into training and test data. Then the difference between the fit on training data and test data measures the model's ability to generalize.

### splitting data
When splitting, we can:

 - split with a fixed %
 - k-fold cross-validation (select k folds without replace and train on set-k , then test on k . repeat for different chunks)
 - leave-one-out cross validation (special case of previous method with k=1)
 - bootstrap (generate new training sets by sampling with replacement)

### bagging
bagging consist of training a lot of weak models and use them together to get a strong model.

 1. draw N bootstrap samples
 2. retrain the model on each sample
 3. average the results ( average for regression, majority vote for classification)
It work great for overfit models (it decreases variance without changing bias) but it doesn't help much with underfit/high bias models

### random forest

```
 repeat k times:
  - draw a boostrap sample from the dataset
  - train a decision tree
   - until the tree is maximum size
      - choose next leaf node
      - select m attributes at random from the p available
      - pick the best attribute/split as usual
 - measure out-of-bag error 
   - evaluate against the samples that were not selected in the boostrap
   - provides measures of strength (inverse error rate), coorelation between trees (which increases the forest error rate), and variable importance

finally, make a prediction by majority among the k trees
```

we talked about variable importance. the key idea is if you scramble the values of a variable and the accuracy of your tree doesn't change much, then the variable isn't very important..
Random forests are more difficult to interpret than single trees; understanding variable importance helps
It's a method easy to parallelize 

### nearest neighbor
Nearest neighbor is really simple, it says plot your point in a space and choose the class of the nearest point to it. Note that this algorithm assumes only numeric attributes.

There is also **k-nearest neighbor** which says take the class of the majority of the *k* nearest points. It gives more resilience to noise.
For choosing k :

 - small k --> fast
 - large k --> bias towards popular labers, ignore outliers 

So in order to use this algorithm, we also need to define some *similitude function*. For that, we can use euclidian distance or cosine similarity(measure of difference between the angles).
Euclidian distance problem, disadvantage is that it is sensitive to the number of dimensions. The disadvantage of cosine is that it favors the dominant components

## optimization

### gradient descent

Gradient descent algorithm work like this:

 - express your learning problem in terms of a cost function that should be minimized
 - starting at initial point, sted 'downhill' until you reach a minimum
 - some situations offer a guarantee that the minimum is the global minimum; others don't

For example, if you have a cost function $J(\theta^{(i)})$  In order to calculate the next coefficient $\theta_0^{(i+1)} <- \theta_0^{(i)} + \alpha \frac{\partial}{\partial \theta_0}J(\theta^{(i)})$ 

$\alpha$ being the learning rate, usually quite small (ex: 0.001)

One problem of gradient descent is that it might go to a local minimum.

### cost functions
Some known cost functions are:

- logistic regression $J(\theta) = \frac{1}{n}\sum$_{i=0}^{n} \log_2(1+exp(-y_i(\theta.x_i))) + \frac{\lamda}{2}\|\theta\|^2$
 - Support Vector Machines (SVM) $J(\theta) = \frac{1}{n}\sum$_{i=0}^{n} \max(1-y_i(\theta.x_i),0) + \frac{\lamda}{2}\|\theta\|^2$
with $\theta$ vector of weights, $x_i$ vector of instance data, and $\frac{\lamda}{2}\|\theta\|^2$ the regularization term

#### regularization term
When we have a lot of weights, it is likely that many are correlated (for examples pixels in a image) So as one weight goes up, another goes down to compsensate. It may lead to verfittin as weights explode. A solution to avoid this is to enforce some condition on the weights to prefer simple models. The regularization term provide this balance. 
It is often based on norm, for example:

 - **lasso** $\frac{\lamda}{2}\bar\theta\bar_1$ with $\bar\theta\bar_1 = \sum_i\bar\theta_i\bar$ , L1 norm
 - **ridge regression** $\frac{\lamda}{2}\bar\theta\bar_2^2$ with $\bar\theta\bar_2^2 = \sqrt{\sum_i \theta_i^2}$

### gradient descent v2
optimized version of gradian descent are:

 - *stochastic gradient descent* : at each step, pick one random data point then continue as if your entire dataset was just the one point. It is quite faster but require more iterations to converge!
 - *minibatch gradient descent* : at each step, pick a small subset of data points then continue as if your entire dataset was just the one point. It is quite faster and more precise than stochastic version!
 - *parallel stochastic gradient descent*: in each of k threads, pick a random data point, compute the gradient and update the weights. weights will be mixed

## unsupervized learning
Zoubin Ghahramani said in 2004:

> almost all work in unsupervised learning can be viewed in terms of learning a probabilistic model of the data

### clustering

 - no precise definition of a cluster
 - output is always a set of sets of items
 - items may be points in some multi-dimensional space (find similar)
 - items may be vertices in a graph (community detection in social networks)

#### k-means clustering
 create clusters by computing the distance between points and the center clusters and assign points to the nearest cluster. then, move the center of the cluster to the center of points belonging to the cluster and iterate.
It can be parallelized using map-reduce (map : assign points to cluster, reduce compute new cluster location)

weaknesses of this algorithm are:

 - you have to know the number of clusters
 - there is sensitivity to initial starting point (no unique solution)
 - there is some sensitivity to stopping threshold

### DBSCAN
 given a dimensional space, group points that are separated by no more than a distance of no more than some epsilon. It has quite some advantages:
 no need to assume a fixed number of clusters, no dependence on starting conditions, only two parameters distance threshold and minimum neighbors, amenable to spatial indexing techniques: O(n log n)
Disadvantages aire that it is sensitive to Euclidian distance measure problems and variable density can be a problem.




## Feature selection
In machine learning and statistics, **feature selection**, is the process of selecting a subset of relevant features (variables, predictors) for use in model construction. Feature selection techniques are used for three reasons:

 - simplification of models to make them easier to interpret by researchers/users
 - shorter training times
 - enhanced generalization by reducing overfitting(formally, reduction of variance)

The central premise when using a feature selection technique is that the data contains many features that are either *redundant* or *irrelevant*, and can thus be removed without incurring much loss of information.

The simplest algorithm is to test each possible subset of features finding the one which minimizes the error rate.  This is an exhaustive search of the space, and is computationally intractable for all but the smallest of feature sets. The choice of evaluation metric heavily influences the algorithm, and it is these evaluation metrics which distinguish between the three main categories of feature selection algorithms: **wrappers**, **filters**, and **embedded methods**.

**Wrapper** methods use a predictive model to score feature subsets. Each new subset is used to train a model, which is tested on a hold-out set. Counting the number of mistakes made on that hold-out set (the error rate of the model) gives the score for that subset. As wrapper methods train a new model for each subset, they are very *computationally intensive*, but usually provide the *best performing feature set* for that particular type of model.

**Filter methods** use a proxy measure instead of the error rate to score a feature subset. This measure is chosen to be *fast to compute*, while still capturing the usefulness of the feature set. Common measures include the **mutual information**, **the pointwise mutual information**, **Pearson product-moment correlation coefficient**, inter/intra class distance or the scores of significance tests for each class/feature combinations.
Filters are usually *less computationally intensive* than wrappers, but they produce a *feature set which is not tuned to a specific type of predictive model*. This lack of tuning means a feature set from a filter is more general than the set from a wrapper, usually giving *lower prediction performance than a wrapper*. However the feature set doesn't contain the assumptions of a prediction model, and so is more useful for exposing the relationships between the features. Many filters provide a feature ranking rather than an explicit best feature subset, and the cut off point in the ranking is chosen via cross-validation. Filter methods have also been used as a preprocessing step for wrapper methods, allowing a wrapper to be used on larger problems.

**Embedded methods** are a catch-all group of techniques which perform feature selection as part of the model construction process. The exemplar of this approach is the *LASSO method* for constructing a linear model, which penalizes the regression coefficients with an L1 penalty, shrinking many of them to zero. Any features which have non-zero regression coefficients are 'selected' by the LASSO algorithm. Improvements to the LASSO include *Bolasso* which bootstraps samples, and *FeaLect* which scores all the features based on combinatorial analysis of regression coefficients. One other popular approach is the *Recursive Feature Elimination algorithm*, commonly used with Support Vector Machines to repeatedly construct a model and remove features with low weights. These approaches tend to be *between filters and wrappers in terms of computational complexity*.

In traditional statistics, the *most popular* form of feature selection is **stepwise regression**, which is a wrapper technique. It is a greedy algorithm that adds the best feature (or deletes the worst feature) at each round. The main control issue is deciding when to stop the algorithm. In machine learning, this is typically done by cross-validation. In statistics, some criteria are optimized. This leads to the inherent problem of nesting. More robust methods have been explored, such as branch and bound and piecewise linear network.


### Correlation feature selection
The **Correlation Feature Selection (CFS)** measure evaluates subsets of features on the basis of the following hypothesis: "Good feature subsets contain features highly correlated with the classification, yet uncorrelated to each other".

The following equation gives the merit of a feature subset S consisting of k features: $Merit_{S_{k}}={\frac {k{\overline {r_{cf}}}}{\sqrt {k+k(k-1){\overline {r_{ff}}}}}}.$

Here,  ${\overline {r_{cf}}}$ is the average value of all feature-classification correlations, and ${\overline {r_{ff}}}$  is the average value of all feature-feature correlations.The CFS criterion is defined as follows:

$CFS =\max _{S_{k}}\left[{\frac {r_{cf_{1}}+r_{cf_{2}}+\cdots +r_{cf_{k}}}{\sqrt {k+2(r_{f_{1}f_{2}}+\cdots +r_{f_{i}f_{j}}+\cdots +r_{f_{k}f_{1}})}}}\right].$

The $r_{cf_{i}}  and $r_{f_{i}f_{j}}$  variables are referred to as correlations, but are not necessarily Pearson's correlation coefficient or Spearman's ρ. Dr. Mark Hall's dissertation uses neither of these, but uses three different measures of relatedness, minimum description length (MDL), symmetrical uncertainty, and relief.

Let $x_i$ be the set membership indicator function for feature $f_i$ ; then the above can be rewritten as an optimization problem:

$CFS =\max _{x\in \{0,1\}^{n}}\left[{\frac {(\sum _{i=1}^{n}a_{i}x_{i})^{2}}{\sum _{i=1}^{n}x_{i}+\sum _{i\neq j}2b_{ij}x_{i}x_{j}}}\right].$

The combinatorial problems above are, in fact, mixed 0–1 linear programming problems that can be solved by using branch-and-bound algorithms.



### stepwise regression
stepwise regression includes regression models in which the choice of predictive variables is carried out by an automatic procedure.  Usually, this takes the form of a sequence of F-tests or t-tests. The main approaches are:

 - **Forward selection**, which involves starting with no variables in the model, testing the addition of each variable using a chosen model comparison criterion, adding the variable (if any) that improves the model the most, and repeating this process until none improves the model.
 - **Backward elimination**, which involves starting with all candidate variables, testing the deletion of each variable using a chosen model comparison criterion, deleting the variable (if any) that improves the model the most by being deleted, and repeating this process until no further improvement is possible.
 - **Bidirectional elimination**, a combination of the above, testing at each step for variables to be included or excluded.

#### Selection criterion
One of the main issues with stepwise regression is that it searches a large space of possible models. Hence it is *prone to overfitting* the data. This problem can be mitigated if the criterion for adding (or deleting) a variable is stiff enough. The key line in the sand is at what can be thought of as the *Bonferroni point*: namely how significant the best spurious variable should be based on chance alone. On a t-statistic scale, this occurs at about ${\sqrt {2\log p}}$ , where $p$ is the number of predictors. Unfortunately, this means that many variables which actually carry signal will not be included. This fence turns out to be the right trade-off between over-fitting and missing signal. If we look at the risk of different cutoffs, then using this bound will be within a $2logp$ factor of the best possible risk. Any other cutoff will end up having a larger such risk inflation.

####Criticism
Stepwise regression procedures are used in data mining, but are controversial. Several points of criticism have been made.

 - When estimating the degrees of freedom, the number of the candidate independent variables from the best fit selected is smaller than the total number of final model variables, causing the fit to appear better than it is when adjusting the r2 value for the number of degrees of freedom. It is important to consider how many degrees of freedom have been used in the entire model, not just count the number of independent variables in the resulting fit.
 - Models that are created may be over-simplifications of the real models of the data.
 
Such criticisms, based upon limitations of the relationship between a model and procedure and data set used to fit it, are usually addressed by verifying the model on an independent data set, as in the **PRESS procedure**.


## Feature extraction
In machine learning, pattern recognition and in image processing, feature extraction starts from an initial set of measured data and builds derived features intended to be informative and non-redundant, facilitating the subsequent learning and generalization steps, and in some cases leading to better human interpretations. Feature extraction is related to dimensionality reduction. General dimensionality reduction techniques includese:

 - Independent component analysis
 - Isomap
 - Kernel PCA
 - Latent semantic analysis
 - Partial least squares
 - Principal component analysis
 - Multifactor dimensionality reduction
 - Nonlinear dimensionality reduction
 - Multilinear Principal Component Analysis
 - Multilinear subspace learning
 - Semidefinite embedding
 - Autoencoder

## References

[wikipedia Feature Selection]{https://en.wikipedia.org/wiki/Feature_selection}
[wikipedia StepWise Regression]{https://en.wikipedia.org/wiki/Stepwise_regression}
[wikipedia Feature Extraction}{https://en.wikipedia.org/wiki/Feature_extraction}
see [scikit clustering]{http://scikit-learn.org/stable/modules/clustering.html} for strenghs and weaknesses of algorithms.
