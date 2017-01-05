---
layout: post
title: "maven Permgen OutOfMemory in scala spark"
categories: debug
tags: debug spark scala permgen maven
---

Trying to increase test coverage on one of my spark scala project using maven to compile, I just encountered a weird error: **java.lang.OutOfMemoryError: PermGen space** . Let's talk a bit about it.
<!--more-->

## the error
It was the end of the year, we were happily adding more test, when suddenly this error appeared on a test class we hadn't touch in a while:

{% highlight shell %}
[32mSparkAvroConverterTest:[0m

[Stage 20:>                                                         (0 + 0) / 2]
[Stage 19:>                                                         (0 + 0) / 2]Exception in thread "Executor task launch worker-0" java.lang.OutOfMemoryError: PermGen space
	at java.lang.ClassLoader.defineClass1(Native Method)
	at java.lang.ClassLoader.defineClass(ClassLoader.java:800)
	at java.security.SecureClassLoader.defineClass(SecureClassLoader.java:142)
	at java.net.URLClassLoader.defineClass(URLClassLoader.java:449)
	at java.net.URLClassLoader.access$100(URLClassLoader.java:71)
	at java.net.URLClassLoader$1.run(URLClassLoader.java:361)
	at java.net.URLClassLoader$1.run(URLClassLoader.java:355)
	at java.security.AccessController.doPrivileged(Native Method)
	at java.net.URLClassLoader.findClass(URLClassLoader.java:354)
	at java.lang.ClassLoader.loadClass(ClassLoader.java:425)
	at sun.misc.Launcher$AppClassLoader.loadClass(Launcher.java:308)
	at java.lang.ClassLoader.loadClass(ClassLoader.java:358)
	at org.apache.spark.executor.Executor$TaskRunner.run(Executor.scala:298)
	at java.util.concurrent.ThreadPoolExecutor.runWorker(ThreadPoolExecutor.java:1145)
	at java.util.concurrent.ThreadPoolExecutor$Worker.run(ThreadPoolExecutor.java:615)
	at java.lang.Thread.run(Thread.java:745)
{% endhighlight %}

## the failed attempts
After a first google search, I found a good explanation of what PermGen is [here](http://stackoverflow.com/questions/1279449/what-is-perm-space) and [here](http://stackoverflow.com/questions/4848669/perm-space-vs-heap-space) with a first possible solution: 
`export MAVEN_OPTS="-Xmx512m -XX:MaxPermSize=128m"` but it didn't work. 

Digging a bit more, my second try was to edit the section about *maven-surefire-plugin* in my *pom.xml*  adding options like forkCount, reuseForks, argLine.
```
<plugin>
  <groupId>org.apache.maven.plugins</groupId>
  <artifactId>maven-surefire-plugin</artifactId>
  <version>2.7</version>
  <configuration>
    <skipTests>false</skipTests>
    <argLine>-Xms2g -Xmx4g -XX:MaxPermSize=512m</argLine>
    <forkCount>1</forkCount>
    <reuseForks>false</reuseForks>
  </configuration>
</plugin>
```
but again it didn't worked and trying to increase MaxPermSize to 1g didn't change anything! Moreover, if I disabled the faulting test, the project was building, and running fine. 

## the debugging
At this point, I thought it was either options to surefire plugin weren't taken into account or I had a memory leak somewhere. To know which one it was, I looked at what was happening in the jvm: My project being named **ti-corp-be**, I did 

{% highlight shell %}
~$ ps -eaf | grep ti-corp-be | grep -v grep | awk '{print $2}'
28069
28072
~$ jmap -heap 28072
Attaching to process ID 28072, please wait...
Debugger attached successfully.
Server compiler detected.
JVM version is 24.65-b04

using thread-local object allocation.
Parallel GC with 4 thread(s)

Heap Configuration:
   MinHeapFreeRatio = 0
   MaxHeapFreeRatio = 100
   MaxHeapSize      = 7990149120 (7620.0MB)
   NewSize          = 1310720 (1.25MB)
   MaxNewSize       = 17592186044415 MB
   OldSize          = 5439488 (5.1875MB)
   NewRatio         = 2
   SurvivorRatio    = 8
   PermSize         = 21757952 (20.75MB)
   MaxPermSize      = 85983232 (82.0MB)
   G1HeapRegionSize = 0 (0.0MB)
.....
{% endhighlight %}

## the solution
And here is is **MaxPermSize      = 85983232 (82.0MB)**  the options wasn't working. Digging a bit more in pom.xml, I then discovered that we were using another plugin *scalatest* with the following configuration:
```
<plugin>
  <groupId>org.scalatest</groupId>
  <artifactId>scalatest-maven-plugin</artifactId>
  <version>1.0</version>
  <configuration>
    <reportsDirectory>${project.build.directory}/surefire-reports</reportsDirectory>
    <junitxml>.</junitxml>
    <filereports>WDF_TestSuite.txt</filereports>
  </configuration>
  ...
```

so I added the line `<argLine>-Xms2g -Xmx4g -XX:MaxPermSize=512m</argLine>` in this plugin configuration, and voila! Tests were working fine, and *jmap* said to me:

{% highlight shell %}
Attaching to process ID 31237, please wait...
Debugger attached successfully.
Server compiler detected.
JVM version is 24.65-b04

using thread-local object allocation.
Parallel GC with 4 thread(s)

Heap Configuration:
   MinHeapFreeRatio = 0
   MaxHeapFreeRatio = 100
   MaxHeapSize      = 4294967296 (4096.0MB)
   NewSize          = 1310720 (1.25MB)
   MaxNewSize       = 17592186044415 MB
   OldSize          = 5439488 (5.1875MB)
   NewRatio         = 2
   SurvivorRatio    = 8
   PermSize         = 21757952 (20.75MB)
   MaxPermSize      = 536870912 (512.0MB)
   G1HeapRegionSize = 0 (0.0MB)
{% endhighlight %}

Looking for references to add, I also found this interesting: [plumbr-permgen-space](https://plumbr.eu/outofmemoryerror/permgen-space). 
