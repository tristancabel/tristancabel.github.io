---
layout: postkatex
title: "dimensionality reduction"
categories: machine_learning
tags: bigdata machine_learning
---

In machine learning and statistics, dimensionality reduction or dimension reduction is the process of reducing the number of random variables under consideration, via obtaining a set of "uncorrelated" principal variables. It can be divided into feature selection and feature extraction.
<!--more-->

Before starting to talk about feature selection, let's introduce 2 mathematical terms we will need:


 - **covariance**: $cov(x,y) = \sum_i (x_i - u_x)(y_i - u_y)$
 - **correlation**: = covariance/standard deivation $corr(x,y) = \frac{cov(x,y)}{\sqrt{\sum_i (x_i - u_x)^2}\sqrt{\sum_i (y_i - u_y)^2}}$


## Feature selection
In machine learning and statistics, **feature selection**, is the process of selecting a subset of relevant features (variables, predictors) for use in model construction. Feature selection techniques are used for three reasons:

 - simplification of models to make them easier to interpret by researchers/users
 - shorter training times
 - enhanced generalization by reducing overfitting(formally, reduction of variance)

The central premise when using a feature selection technique is that the data contains many features that are either *redundant* or *irrelevant*, and can thus be removed without incurring much loss of information.

The simplest algorithm is to test each possible subset of features finding the one which minimizes the error rate.  This is an exhaustive search of the space, and is computationally intractable for all but the smallest of feature sets. The choice of evaluation metric heavily influences the algorithm, and it is these evaluation metrics which distinguish between the three main categories of feature selection algorithms: **wrappers**, **filters**, and **embedded methods**.

 - **Wrapper** methods use a predictive model to score feature subsets. Each new subset is used to train a model, which is tested on a hold-out set. Counting the number of mistakes made on that hold-out set (the error rate of the model) gives the score for that subset. As wrapper methods train a new model for each subset, they are very *computationally intensive*, but usually provide the *best performing feature set* for that particular type of model.
 - **Filter methods** use a proxy measure instead of the error rate to score a feature subset. This measure is chosen to be *fast to compute*, while still capturing the usefulness of the feature set. Common measures include the **mutual information**, **the pointwise mutual information**, **Pearson product-moment correlation coefficient**, inter/intra class distance or the scores of significance tests for each class/feature combinations. Filters are usually *less computationally intensive* than wrappers, but they produce a *feature set which is not tuned to a specific type of predictive model*. This lack of tuning means a feature set from a filter is more general than the set from a wrapper, usually giving *lower prediction performance than a wrapper*. However the feature set doesn't contain the assumptions of a prediction model, and so is more useful for exposing the relationships between the features. Many filters provide a feature ranking rather than an explicit best feature subset, and the cut off point in the ranking is chosen via cross-validation. Filter methods have also been used as a preprocessing step for wrapper methods, allowing a wrapper to be used on larger problems.
 - **Embedded methods** are a catch-all group of techniques which perform feature selection as part of the model construction process. The exemplar of this approach is the *LASSO method* for constructing a linear model, which penalizes the regression coefficients with an L1 penalty, shrinking many of them to zero. Any features which have non-zero regression coefficients are 'selected' by the LASSO algorithm. Improvements to the LASSO include *Bolasso* which bootstraps samples, and *FeaLect* which scores all the features based on combinatorial analysis of regression coefficients. One other popular approach is the *Recursive Feature Elimination algorithm*, commonly used with Support Vector Machines to repeatedly construct a model and remove features with low weights. These approaches tend to be *between filters and wrappers in terms of computational complexity*.

In traditional statistics, the *most popular* form of feature selection is **stepwise regression**, which is a wrapper technique. It is a greedy algorithm that adds the best feature (or deletes the worst feature) at each round. The main control issue is deciding when to stop the algorithm. In machine learning, this is typically done by cross-validation. In statistics, some criteria are optimized. This leads to the inherent problem of nesting. More robust methods have been explored, such as branch and bound and piecewise linear network.


### Correlation feature selection
The **Correlation Feature Selection (CFS)** measure evaluates subsets of features on the basis of the following hypothesis: "Good feature subsets contain features highly correlated with the classification, yet uncorrelated to each other".

The following equation gives the merit of a feature subset S consisting of k features: $Merit_{S_{k}}={\frac {k{\overline {r_{cf}}}}{\sqrt {k+k(k-1){\overline {r_{ff}}}}}}.$

Here,  ${\overline {r_{cf}}}$ is the average value of all feature-classification correlations, and ${\overline {r_{ff}}}$  is the average value of all feature-feature correlations.The CFS criterion is defined as follows:

$CFS =\max\limits_{S_{k}} \Big[ \frac{r_{cf_{1}}+r_{cf_{2}}+\cdots +r_{cf_{k}}}{\sqrt{k + 2(r_{f_{1}f_{2}} + \cdots  + r_{f_{i}f_{j}} + \cdots + r_{f_{k}f_{1}} )}} \Big]$ 

The $r_{cf_{i}}  and $r_{f_{i}f_{j}}$  variables are referred to as correlations, but are not necessarily Pearson's correlation coefficient or Spearman's ρ. Dr. Mark Hall's dissertation uses neither of these, but uses three different measures of relatedness, minimum description length (MDL), symmetrical uncertainty, and relief.

Let $x_i$ be the set membership indicator function for feature $f_i$ ; then the above can be rewritten as an optimization problem:

$CFS =\max\limits_{x\in (0,1)^{n}} \Big[ \frac{(\sum_{i=1}^{n}a_{i}x_{i})^{2}}{\sum_{i=1}^{n}x_{i} +\sum_{i\neq j}2b_{ij}x_{i}x_{j}} \Big]$

The combinatorial problems above are, in fact, mixed 0–1 linear programming problems that can be solved by using branch-and-bound algorithms.



### stepwise regression
stepwise regression includes regression models in which the choice of predictive variables is carried out by an automatic procedure.  Usually, this takes the form of a sequence of F-tests or t-tests. The main approaches are:

 - **Forward selection**, which involves starting with no variables in the model, testing the addition of each variable using a chosen model comparison criterion, adding the variable (if any) that improves the model the most, and repeating this process until none improves the model.
 - **Backward elimination**, which involves starting with all candidate variables, testing the deletion of each variable using a chosen model comparison criterion, deleting the variable (if any) that improves the model the most by being deleted, and repeating this process until no further improvement is possible.
 - **Bidirectional elimination**, a combination of the above, testing at each step for variables to be included or excluded.

#### Selection criterion
One of the main issues with stepwise regression is that it searches a large space of possible models. Hence it is *prone to overfitting* the data. This problem can be mitigated if the criterion for adding (or deleting) a variable is stiff enough. The key line in the sand is at what can be thought of as the *Bonferroni point*: namely how significant the best spurious variable should be based on chance alone. On a t-statistic scale, this occurs at about ${\sqrt {2\log p}}$ , where $p$ is the number of predictors. Unfortunately, this means that many variables which actually carry signal will not be included. This fence turns out to be the right trade-off between over-fitting and missing signal. If we look at the risk of different cutoffs, then using this bound will be within a $2logp$ factor of the best possible risk. Any other cutoff will end up having a larger such risk inflation.

#### Criticism
Stepwise regression procedures are used in data mining, but are controversial. Several points of criticism have been made.

 - When estimating the degrees of freedom, the number of the candidate independent variables from the best fit selected is smaller than the total number of final model variables, causing the fit to appear better than it is when adjusting the r2 value for the number of degrees of freedom. It is important to consider how many degrees of freedom have been used in the entire model, not just count the number of independent variables in the resulting fit.
 - Models that are created may be over-simplifications of the real models of the data.
 
Such criticisms, based upon limitations of the relationship between a model and procedure and data set used to fit it, are usually addressed by verifying the model on an independent data set, as in the **PRESS procedure**.


## Feature extraction
In machine learning, pattern recognition and in image processing, feature extraction starts from an initial set of measured data and builds derived features intended to be informative and non-redundant, facilitating the subsequent learning and generalization steps, and in some cases leading to better human interpretations. Feature extraction is related to dimensionality reduction. General dimensionality reduction techniques includes:

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

## adding dimensions
Some time, we may want to do the opposite and had more dimensions. To do so, we can:

 - add cross-product terms $\phi(1,x_1,x_2) = (1,x_1, x_2, x_1^2, x_2^2, x_1 x_2)$

## References

 - [wikipedia dimensionality reduction](https://en.wikipedia.org/wiki/Dimensionality_reduction)
 - [wikipedia Feature Selection](https://en.wikipedia.org/wiki/Feature_selection)
 - [wikipedia StepWise Regression](https://en.wikipedia.org/wiki/Stepwise_regression)
 - [wikipedia Feature Extraction](https://en.wikipedia.org/wiki/Feature_extraction)
