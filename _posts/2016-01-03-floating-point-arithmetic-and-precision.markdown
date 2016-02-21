---
layout: post
title: "floating point arithmetic and precision"
categories: archive
tags: archive precision
---

In this post, I want to talk a bit about floating-point arithmetic and precision applied to GPUs!

<!--more-->

### IEEE 754 introduction
Let's start by talking a bit about how do we represent floating-point numbers in digital world. These numbers are defined by the IEEE Standard for Floating-Point Arithmetic (IEEE 754) and as wikipedia says:

>it's a technical standard for floating-point computation established in 1985 by the Institute of Electrical and Electronics Engineers (IEEE)." The current version is IEEE 754-2008. 

In this standard, finite numbers are represented by 3 integers : 

  * *s* a sign
  * *e* an exponant
  * *f* a fraction 

which leads to the numerical value : `(-1)exp(s) * f * b*exp(e)` . *b* being the base which is either 2 or 10.

In a software, we principally use 2 types of floating point numbers, both in base 2 : 

  * single(float) (32 bits : s=1bit , e=8bits, c=23bits)
  * double        (64 bits : s=1bit, e=11bits, c=52bits) 

There is also some special numbers like *NaN* (Not a Number) or +/- infinity but I won't talk about it here.

### Rounding rules
Here come the interesant part! The standard defines four rounding rules. 

  * **rn** : Round to nearest, ties to even – rounds to the nearest value; if the number falls midway it is rounded to the nearest value with an even (zero) least significant bit, which occurs 50% of the time; this is the default for binary floating-point and the recommended default for decimal.
  * **rz** : Round toward 0 – directed rounding towards zero (also known as truncation).
  * **ru** : Round toward +∞ – directed rounding towards positive infinity (also known as rounding up or ceiling).
  * **rd** : Round toward −∞ – directed rounding towards negative infinity (also known as rounding down or floor).

With this, the IEEE 754 standard defines operations like addition, substraction, multiplication, division with their precision. This standard allows us to expect having the same results if we do an operation on a CPU or on another processor which is essential.

### The Fused Multiply-Add (FMA)
 Another common operation defined in the standard is the FMA operation. It was introduced in the 2008 revision of IEEE 754 standard, it computes `rn(a*b+c)` and what's interesting is that there is only 1 rounding step! Without it, we would have to compute `rn(rn(a*b) +c)` with 2 rounding steps which is different!

Let's take an example and try to compute `x*exp(2)-1` with *x = 1.0008* and 4 digits of precision. the correct result is `1.60064*10exp(-4) = 1.6006*10exp(-4)` if we uses 4 digits of precision.
However, if we do `rn(rn(x*x) - 1)`, we have `rn(rn(1.00160064) - 1) = rn(1.0016 -1) = 1.6*10exp(-4)` so we lose quite some precision!
 

### Precision in GPU programming
Back in 2009, where CUDA was still something relatively new, I encounter a problem with the division! Indeed, the first Nvidia cards with CUDA ( compute capability 1.2 and below), had a precision problem with the division. This operation was non IEEE 754 compliant (maximum ulp errors 2) which leads to different results for an algorithm executed on a GPU or on a CPU.

But what's really interesting is that nowadays, the story repeat itself! There is a problem of non IEEE-754 rounding rule compliance with some mobile phone GPUs as it is stated in this interesting article : [arm blog](http://community.arm.com/groups/arm-mali-graphics/blog/2013/06/11/benchmarking-floating-point-precision-in-mobile-gpus--part-ii)

### Conclusion
So if you are going to use a uncommon device to do some floating-point operation, don't forget to read the documentation and look for the IEEE-754 compliance. Otherwise, good luck for debugging!