---
layout: postkatex
title: "data visualisation"
categories: bigdata
tags: bigdata visualisation 
---

fourth part of notes of Coursera big data specialization.
<!--more-->

Visualisation is important as it can :

 - reveal patterns statistics may not
 - allow human to comprehend patterns in larger datasets than they could otherwise
 - exercices the highest bandwidth channel into the humain brain

### Data Types and Visual Mappings
Data types can be classified as Nominal, Ordinal or Quantitative

  - **Nominal** : fruits (apples, oranges) . Operations : =, ne
  - **Ordered** : quality of meats ( B, A, AA, AAA) . Operations : =, ne, <, >, <=, >=, -
  - **Quantitative** : Interval(arbitrary location of zero) such as Dates, location  or Ratio (zero fixed) like physical measurement. Operations : =, ne, <, >, <=, >=, -, /

Bertin in *Semiology of Graphics* in 1967 defined 7 Visual attributes : **Position, Size, Value, Texture, Color, Orientation and Shape**. (W1 L1 04 -slides 31).
Bertin also said that variable types map to certain types of attributes

 |Position    | N | O   |  Q  |
 |Size        | N | O   |  Q  |
 |Value       | N | O   | (q) | 
 |Texture     | N | (o) |     |
 |Color       | N |     |     |
 |Orientation | N |     |     |
 |Shape       | N |     |     |

Moreover, the Human brain perceptual properties follow the according list ( from more accurate to less accurate) : *Position, Length, Angle/Slope , Area, Volume, Color/Density* . 

When mapping data to visual attributes, the challenge is to pick the best encoding( or mapping) from nay possibilities, consider :

 - **importance Ordering** : encode the most important information in th emost perceptually accurate wat
 - **Expressiveness** : Depict all the *data*, and *only* the data.
 - **Consistency** : The properties of the image (visual attributes) should match the properties of the data

Humans tend to be better at estimating length than either area or volume! Usually, length is estimated from 0.9 to 1.1, area from 0.6 to 0.9 and volume from 0.5 to 0.8. For example for a circle or area A, the human eye will perceive it as a circle of area $S=0.98*A^0.87$ J. J. Flannety, 1971, "the relative effectiveness of some graduated point symbols in the presentation of quantitative data..

To critique a visualization, do the following : First, consider the purpose of the visu and who the intended audience is. Then, ascertain your initial reaction. Then, examine the visu in detail. Then, answer questions like : is the design appealing? is it immediately understandable? does is require excessive cognitive effort? Does it successfully highlight important info, does it omit important info? does it distort the info? is it memorable?.

### Ethics and Privacy
Be very carefull about therepresentation of the test  set if we want to do generalisation.
In the "barrow alcoholism study", participants were not in control of their data nor the context in which they were presented.
Even those who did not participate in the study incurred harm. We have to *smell* the ethics. Researchers appear to have placed their own interests ahead of those of the research subjects, the client and society. Break *ethical principles*.

There is reponsabilities for Funders, Clients, Employers, Research Subjects, Research Team colleagues, Society.
We have to be carefull protecting sensitive data.

You have to be carefull about security and protecting privacy. Access control, query control, perturbation-based techniques, secure multi-party computation, anonymity.
Removing "identifiers" may not be sufficient to ensure anonymity. k-anonymity ensure that for any query, each record cannot be distinguished from at least k-1 other records. 
Two recurring features of privacy failures are :

 - high-dimensional data to distinguish individuals
 - combination of multiple datasets to re-identify individuals 

#### Differential Privacy

 - Differential privacy is a guarantee from the data collector to the individuals in the dataset.
 - The chance that the noisy, publicized result will be *R* is about the same whether or not you include your data:
 (Pr(Q(D<sub>I</sub>) = R))/(Pr(Q(D<sub>I/plusequal i</sub>) = R)) /infequal A for all I,i,R
 - Q is the privatized query
 - A is a value very close to 1. If A is large, no privacy is provided. If A=1, then the data has no utility
 - we define A = e<sup>/sigma</sup> for some small /sigma > 0


#### Laplacian noise
 the sensitivity is /delta F . We know we must add noise that is a function of this sensitivity and the higher the sensitivity, the more we have to change the result to prevent an adversary from distinguishing between the two possible worlds. The simplest approach is Laplacian noise!

f(x|/mu,b) = 1/(2b) e<sup>(- <abs>x-/mu</abs>/b)</sup> is Lap(/mu,b) where 
 - /mu is the position
 - b is the scale (the spread)

D is the dataset
The scale depend on /sigma, /lambda F , b = /delta F / /epsilon
The posision should depend on F(D), /mu = F(D)

So privatized results R = F(D1) + Lap(/delta F/ /sigma)

#### problems in differential privacy

 - It's best for low-sensitivity queries. For example sums or max are problems. Consider the question "what is the total income earned by men vs women?" . A single very-high income individual could change the result by an enormous amount, so you'd have to add a lot of noise for this worst-case individual. 
 - adaptive querying exposes more privacy: if I ask a series of differentially private queries, I can disambiguate between possible worlds. So we have to add *even more* noise to the results but doing so, we may destroy the utility of the output.

## Reproducibility
The claim needs to be independently verified. Cloud computing can be an appropriate platfom for reproductible data science as it includes "Code + data + Environment". Otherwise, there is also virtual machines and containers.

## Cloud computing
Just like generators of factories from the 19th century had become unnecessary in the 20th century, the same is happening nowadays with servers. Most companies have their own and might switch to cloud (provided that issues such as privacy and security are solved).

There is **infrastructure-aas**, **platform-aas**, **software-aas**.
Specific advantages of cloud for research is:

 - burst capacity (1k cores for 1 day costs the same (or less) thatn 1 core dor 1000 days).
 - reproductibility as investigators use the same tools and data.
 - sharing and collaboration
 - eliminate redundant infrastructure across units


## Costs and sharing
who pays for reproductibility? Answer: you, you or them, them. codes for hosting data shouldn't be neglected.

Observation on Big Data : the **only** solution is to push the computation to the data, rather than push the data to the computation! It takes days to transfer 1TB over the internet, copying a petabyte is operationally impossible.