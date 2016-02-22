---
layout: post
title: "Heisenbug bug"
categories: debug
tags: bug debug
---
As stated in [Wikipedia](https://en.wikipedia.org/wiki/Heisenbug),

> In computer programming jargon, a heisenbug is a software bug that seems to disappear or alter its behavior when one attempts to study it .

I recently came across one whilst developing a small program:

<!--more-->

- when I tried to run this program from the console, it crashed with a *segmentation fault*

{% highlight bash %}
DEBUG - Tue Feb 16 12:39:29 2016 - creation of mesher 
DEBUG - Tue Feb 16 12:39:29 2016 - mesher run 
DEBUG - Tue Feb 16 12:39:29 2016 - run  start reading mesh  "/home/tcabel/Devel/tracesinria/Tests/Regr/SAT_HY_1.1/Result/Data_A/tmp_hydro_1.1" 
INFO  - Tue Feb 16 12:39:29 2016 - numTracesMesher : elem_type = 20 
DEBUG - Tue Feb 16 12:39:29 2016 - mesh dimension : 2 
INFO  - Tue Feb 16 12:39:29 2016 - nb points =  2601  nb_cells =  2500  nb_faces_or_aretes =  5100 
DEBUG - Tue Feb 16 12:39:29 2016 - vertexes and points creation 
DEBUG - Tue Feb 16 12:39:29 2016 - cells creation 
DEBUG - Tue Feb 16 12:39:29 2016 - mesh read finished, setting cells 
Segmentation fault (core dumped)
{% endhighlight %}

 - when I tried to run the same program with the same arguments under **gdb**, it worked fine

{% highlight bash %}
DEBUG - Tue Feb 16 12:35:34 2016 - creation of mesher 
DEBUG - Tue Feb 16 12:35:34 2016 - mesher run 
DEBUG - Tue Feb 16 12:35:34 2016 - run  start reading mesh  "/home/tcabel/Devel/tracesinria/Tests/Regr/SAT_HY_1.1/Result/Data_A/tmp_hydro_1.1" 
INFO  - Tue Feb 16 12:35:34 2016 - numTracesMesher : elem_type = 20 
DEBUG - Tue Feb 16 12:35:34 2016 - mesh dimension : 2 
INFO  - Tue Feb 16 12:35:34 2016 - nb points =  2601  nb_cells =  2500  nb_faces_or_aretes =  5100 
DEBUG - Tue Feb 16 12:35:34 2016 - vertexes and points creation 
DEBUG - Tue Feb 16 12:35:34 2016 - cells creation 
DEBUG - Tue Feb 16 12:35:34 2016 - mesh read finished, setting cells 
DEBUG - Tue Feb 16 12:35:34 2016 - finished setting cells 
INFO  - Tue Feb 16 12:35:34 2016 - Mesh read in "00:00:00.010" 
DEBUG - Tue Feb 16 12:35:34 2016 - run  mesher finisher  
DEBUG - Tue Feb 16 12:35:34 2016 - creation of meshView 
DEBUG - Tue Feb 16 12:35:34 2016 - meshView finished 
{% endhighlight %}

One way to debug this type of error is to dump a **core file** and use it to get the memory state from the crash. In order to get this core file, follow the below:

1. `ulimit -c unlimited`
2. relaunch the program which will now dump a core file when it crashes.
3. relaunch gdb with this core file `gdb ./program core.XXX`

In my case, it was an incorrect array index that was causing this error.

When you finish debugging, don't forget to execute `ulimit -c 0` to disable the dump of core files each time a program crashes.
