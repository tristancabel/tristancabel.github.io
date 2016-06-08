---
layout: postkatex
title: "predictive analytics - statistical interference"
categories: bigdata
tags: bigdata machine_learning statistics
---

Second part of notes of Coursera big data specialization.
<!--more-->

First, let's introduce the curse of big data as defined by *Vincent Granville*:

> The curse of big data is the fact that when you search for patterms in very, very large data sets with billions or trillions of data points and thousands of metrics, you are bound to identify coincidences that have predictive power.

statistical interference is a method for drawing conclusions about a population from sample data. It has two key methods: **Hypothesis tests** (significance tests) and **confidence intervals**. Hypothesis testing is comparing an experimental group and a control group. Studying the probability of seeing data given the null hypothesis is called the **frequentist approach to statistics** as opposed to the **bayesian approach to statistics**.


# Null Hypothesis testing

Let's define:

 - $H_0$ Null hypothesis = no differences between the groups
 - $H_A$ Alternative hypothesis = statistically significant difference between the groups. "difference" defined in terns of some **test statistic**.

<table>
  <thead>
    <tr>
      <td> </td>
      <td> Do not reject $H_0$ </td>
      <td> Reject $H_0$ </td>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td> $H_0$ is true </td>
      <td> <em>Correct decision</em> $1 - \alpha$ </td>
      <td> <em>Type 1 error</em>  $\alpha$ </td>      
    </tr>
    <tr>
      <td> $H_0$ is false </td>
      <td> <em>Type 2 error</em> $\beta$ </td>
      <td> <em>Correct decision</em> $1 - \beta$ </td>      
    </tr>
  </tbody>
</table>

For the difference, we do not know if the difference in two treatments is not just do to chance but we can calculate the odds that it is! **Statistical significance** is attained when a *p-value* is less than the *significance level* $\alpha$ . The p-value is the probability of *obtaining at least as extreme results given that the null hypothesis is true* whereas the significance level $\alpha$ is the probability of *rejecting the null hypothesis given that it is true* . Or in repeated experiments at this sample size, how often would you see a result at least this extrem assuming the null hypothesis? Or  A result is said to be statistically significant if it allows us to reject the null hypothesis. Usually a **significance level  $\alpha = 0.05$** is used.

The **p-value** is defined as the probability, under the assumption of hypothesis $H$, of obtaining a result equal to or more extreme than what was actually observed. Thus, the smaller the p-value, the larger the significance because it tells the investigator that the hypothesis under consideration may not adequately explain the observation Depending on how it is looked at, the "more extreme than what was actually observed" can mean  $X \geq x$ (right-tail event) or $X \leq x\$ (left-tail event) or the "smaller" of $(X \leq x) \space and \space (X \geq x)$ (double-tailed event). Thus, the **p-value** is given by

 - $\Pr(X \geq x \|H)$ for right tail event,
 - $\Pr(X \leq x \|H)$ for left tail event,



We will now study two methods to ensure results are significant. One analytic(classical method) and one computational. To do this, let's use a test case of cancer treatment, comparing two treatments using mean days of survival.

## Classical method: derive the sampling distribution

 We make the assumptions that:

 - the number of survival days follow a *normal distribution*
 - the variances of the two set of patients are the same (or that the sample size are the same)

Let's construct a **t-statistic** (a ratio of the departure of an estimated parameter from its notional value and its standard error.):  $t_{statistic} = \frac{statistic - hypothetized\ value}{estimated\ standard\ error\ of\ statistic}$

For our case : statistic is diff in mean $ statistic = \bar{T} - \bar{S}$, hypothetized value is 0 and "standard error" is just another term for standard deviation of this sampling distribution. So we want to find, what is the sampling distribution of $\bar{T} - \bar{S}$. It can be described by it-s mean and variance:

 - **mean**: $\mu_{\bar{T} - \bar{S}} = \mu_{\bar{T}} - \mu_{\bar{S}}$
 - **variance**: $\sigma_{\bar{T} - \bar{S}}^2 = \sigma_{\bar{T}}^2 + \sigma_{\bar{S}}^2$

Then, using the *central Limit theorem*,  $\sigma_{\bar{S}}^2 = \frac{\sigma_S^2}{n_S}$ . So for our case,  $\sigma_{\bar{T} - \bar{S}} = \sqrt{\frac{\sigma_T^2}{n_t} + \frac{\sigma_S^2}{n_s}}$  and $t = \frac{\bar{T} - \bar{S}}{\sigma_{\bar{T} - \bar{S}}}$

But, there is many t-distribution (as it is sensitive to sample size), and the particular form of the t distribution is determined by its *degree of freedom*. The degrees of freedom refers to the number of independent observations in a set of data. When estimating a mean score or a proportion from a single sample, the number of independent observations is equal to the sample size minus one. Hence, the distribution of the t statistic from samples of size 8 would be described by a t distribution having 8 - 1 or 7 degrees of freedom. For other applications, the degrees of freedom may be calculated differently.
Lastly, with this *t* and *degree of freedom*, we can use a chart to have the cumulative probability  $P(T \leq t)$.


## Computational method
Instead of formally computing these terms, we can use a computational method such as:

 - **Monte Carlo simulation** "what would happen if we ran this experiment 1000 times?" . --> do a simulated experiment
 - **Resampling** : mix values between test and standard results. Then recompute the means, and the difference in the means. Re do this at least 10000 times. We can then see if the difference in the means in the initial results is significant. The key assumption is independance.
 
To decide when the result can be considered significant, consider the null hypothesis, run the experiment and compute the 5th and 95th percentiles of all the different results.

## effect size
Effect size is used to try to see *how* significant a result is (not only if it is significant or not). It is used prolifically in meta-analysis to combine results from multiple sudies. $ES = \frac{Mean\ of\ experimental\ group - Mean\ of\ control\ group}{standard\ deviation}$

$ES = \frac{\bar{X_1} - \bar{X_2}}{\sigma_{pooled}}$  with $\sigma_{pooled}=\sqrt{\frac{\sigma_1^2(n_1 -1) + \sigma_2^2(n_2 -1) }{(n_1-1 + (n_2-1))}}$

From this, we can say that **a standardized mean difference effect size of 0.2 is small, 0.5 is medium and 0.8 is large**.

## Meta-analysis
From Fisher(1944)

> When a number of quite independant tests of significance have been made, it sometimes happens that although few or none can be claimed individually as significant, yet the aggregate gives an impression that the probabilities are on the whole lower than would often have been obtained by chance.

So it's aggragating results! We can use a weighted average, average across multiple studies, but give more weight to more precise studies. We can define weights has inverse-variance wight $w_i = \frac(1}{se^2}$ (se = standard error) or use a simple method: weight by sample size

## Benford's Law
*Benford's Law* is a tool to detect fraud
An example to intuite how it workds is this:

 - given a sequence of cards labbeled 1, 2, 3, .., 999999
 - put them in a hat, one by one, in order
 - after each card, ask "what is the probability of drawing a card where the first digit is the number 1?"

The first digit of our data should follow the same distribution. However, to use it, data has to span over several orders of magnitude and it can't have min/max cutoffs.


## Multiple hypothesis testing

 - if you perform experiments over and over, you're bound to find something.
 - this is a bit different than the publication bias problem: same sample, different hypotheses
 - significance level must be adjusted down when performing multiple hypothesis tests (not 0.05 but much lower)

Indeed, if $P(detecting\ an\ effect\ when\ there\ is\ none) = \alpha = 0.05$ ,then :

 - $P(detecting\ an\ effect\ when\ it\ exists)=1-\alpha$
 - $P(detecting\ an\ effect\ when\ it\ exists\ on\ every\ experiment)=(1-\alpha)^k$
 - $P(detecting\ an\ effect\ when\ there\ is\ none\ on\ at\ least\ one\ experiment) = 1-(1-\alpha)^k$

Solutions of familywise error rate corrections are (sidak being the more conservative):

 - *Bonferroni Correction* $\alpha_c = \frac{\alpha}{k}$
 - *Sidak Correction* $\alpha=1-(1-\alpha_c)^k$ with $\alpha_c=1-(1-\alpha)^{\frac{1}{k}}$

Another less conservative solution is considering the **false discovery rate**  $FDR= Q = \frac{FD}{FD + TD}$


# Bayesian approach

Differences between Bayesians and frequentist in what is fixed are:

 - **Frequentist** Data are a repeatable random sample (there is a frequency), underlying parameters remain constant during this repeatable process, *parameters are fixed*
 - **bayesian** Data are observed from the realized sample, parameters are unknown and described praobabilistically and *data are fixed*.

The bayesian approach is studying the probability of a given outcome, given this data: $P(H \vert D) =  \frac{P(D \vert H) \times P(H)}{P(D)}$ .
This gives a key benefit and a key weakness: *the ability to incorporate prior knowledge, and the need to incoporate prior knowledge.*

terms of bayesian theorem can be express like this:

 - $P(H \vert D)$ _posterior_ the probability of our hypothesis being true given the data collected
 - $P(D \vert H)$ _likelihood_ probability of collecting this data when our hypothesis is true
 - $P(H)$ _prior_ the probability of the hypothesis being true before collecting data
 - $P(D)$ _marginal_ what is the probability of collecting this data under all possible hypotheses

As an example of bayesian approach, let's solve this problem:

 - 1% of women at age forty who participate in routine screening have reast cancer
 - 80% of women with breast cancer will get positive mammographies
 - 9.6% of women without breast cancer will also get positive mammographies
 - A women in this age group had a positive mammography in a routine screening
 
What is the probability that she actually has breast cancer?

$P(positive) = P(positive\|cancer) \times P(cancer) + P(positive\|no_cancer) \times P(no_cancer) = 0.8 \*0.01 + 0.096 \* 0.99 = 0.103$
$P(cancer\|positive) = frac{P(positive\|cancer) \times (P(cancer)}{P(positive)} = \frac{0.8\*0.01}{0.103} = 0.78 = 78\%$
