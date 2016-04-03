---
layout: post
title: "COURSERA Data Science, SF crime analysis"
categories: coursera
tags: coursera python visualization
---


In this assignment, we are trying to extract relevant data from the crimes occured in San Francisco durring the summer 2014. This is an exercise from a coursera lesson.

<!--more-->


We will find that crimes occured mostly during the evening and that most belong to the category **LARCENY/THEFT**. Furthermore, we will show that this type of crime is concentrated in a certain neighborhood.

### All Crimes 
To start, let's look at the time distribution of all the crimes of a week. For the first following figure, on the x-axis, we define a variable *HourOfWeek* as this : *Monday 00h00 = 0, Monday 01h00 = 1 , Monday 13h00 = 12 , Tuesday 00h00 = 24,* ...). For the second following figure, we group by Hour of a day.

<img src="{{ site.url }}/public/6_crime_vis/crimes_distribution_per_hour.png" width="900">

we can clearly see that most crimes occured during the afternoon/evening. I also expected most crimes to occur during the week end but the data show that it's not so obvious.

<img src="{{ site.url }}/public/6_crime_vis/crimes_per_day.png" width="900">

There is just a slight increase of crimes during the week end.


We can also plot crimes on a map to determine whether some neighborhoods within San Francisco are more dangerous than others.

<img src="{{ site.url }}/public/6_crime_vis/map_all_crimes.png" width="1024">

Indeed, this figure demonstrates that crimes occured mostly in the central areas of SF.


Now, let's look at the top crime categories

<img src="{{ site.url }}/public/6_crime_vis/category.png" width="1024">

We can see that the top crime here is **LARCENY/THEFT** by a huge margin so let's focus on this category of crimes.

### Larceny/Theft crimes

First, let's see in which areas of SF this category of crime occured the most.

<img src="{{ site.url }}/public/6_crime_vis/map_larceny.png" width="1024">

We can observe that this crime is highly concentrated in one neighborhood, even more that the first map (containing all the crimes).

Now let's see if this type of crime is occuring more often at specific times of the day and if it is distributed evenly during the week:
<img src="{{ site.url }}/public/6_crime_vis/larceny_per_day.png" width="900">

Here, as opposed to when we were considering all the crimes, we can clearly see that there is an increase of *Larceny/Theft* crimes on Fridays and week ends.

<img src="{{ site.url }}/public/6_crime_vis/larceny_per_hour.png" width="900">
Moreover, these last figures tells us that most the crimes occured during the afternoon. Lastly, we notice that there is a peak at lunch time.

For reproductibility, here is the [script]({{site.url}}/public/6_crime_vis/crime_visu.py) used to create my pictures.