---
layout: post
title: "COURSERA Data Science, SF crime analysis"
categories: coursera
tags: coursera python visualization
---


In this assignment, we are trying to analyze crime occured in San Francisco durring the summer 2014. This is an assignment from a coursera lesson.

<!--more-->


We will show that the crimes occur mostly during the evening and that most of them belongs to the category **LARCENY/THEFT**. Moreover, we will show that this type of crime is concentrated in a certain neighboorhood.

### All Crimes 
To start, let's look at the time distribution of all the crimes in a week. In abscisse, we define a variable HourOfWeek as this : *Monday 00h00 = 0, Monday 01h00 = 1 , Monday 13h00 = 12 , Tuesday 00h00 = 24,* ...). 

<img src="{{ site.url }}/public/6_crime_vis/crimes_distribution_per_hour.png" width="900">

we can clearly see that most of the crimes are occuring during the afternoon/evening! However, I expected most crime to occur during the week end, but the following figure show that it's not so obvious.

<img src="{{ site.url }}/public/6_crime_vis/crimes_per_day.png" width="900">

There is just a slight increase of crimes during the week end.


We can also try to see if some neighboorhood are more dangerous than others in San Francisco.

<img src="{{ site.url }}/public/6_crime_vis/map_all_crimes.png" width="1024">

This map show us that crimes are concentrated in a central area with some other spots in the city.


Now, let's look at the top crime categories

<img src="{{ site.url }}/public/6_crime_vis/category.png" width="1024">

We can see that the top crime here is **LARCENY/THEFT** by a huge margin so let's focus on this category of crimes.

### Larceny/Theft crimes

First, let's try to see in which areas is this crime occuring.

<img src="{{ site.url }}/public/6_crime_vis/map_larceny.png" width="1024">

We can observe that this crime is really concentrated in one neighboorhood, even more that the first map (containing all the crimes).

Now let's see if this type of crime is more occuring at some specific times and if it is distributed evenly during the week:
<img src="{{ site.url }}/public/6_crime_vis/larceny_per_day.png" width="900">

<img src="{{ site.url }}/public/6_crime_vis/larceny_per_hour.png" width="900">
On the first figure, as opposed to when we were considering all the crimes, we can clearly see that there is an increase of *Larceny/Theft* crimes during Friday and the Week end. Moreover, the second figure tell us that most of the crimes are occuring during the afternoon. Lastly, we can also notices that there is a decrease of crime everyday just after lunchtime everyday. Maybe it's a nap time effect?

For reproctibility, here is the [script]({{site.url}}/public/6_crime_vis/crime_visu.py) used to create my pictures.