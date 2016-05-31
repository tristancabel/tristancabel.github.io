---
layout: post
title: "data manipulation"
categories: bigdata
tags: bigdata  SQL NoSQL
---

Notes on first lesson of Coursera big data specialization.
<!--more-->

Data science operates over 4 dimensions :

 - **Breadth** : from tool(Hadoop, postreSQL) to abstractions (MapReduce, Relational Algebra)
 - **Depth** : from structures(Management, Relational Algebra) to statistics( Analysis, Linear Algebra)
 - **Scale** : from desktop (main memory, R) to cloud (distributed, Hadoop, S3)
 - **Target** : from hackers( proficiency in python, R) to analysts(little or no programming)

A big data definition could be the one from Michael Franklin:

>Big Data is any data that is expensive to manage and hard to extract value from

Big Data operates on 3 challenges (**3V**) : **Volume**(size), **Velocity**(latency) and **Variety**(diversity of sources).

## Relational Database
A data Model is structured over three components :

 1. Structures(rows and columns? key-value pairs?)
 2. Constraints(all rows must have the same nb of columns, a child cannot have two parents)
 3. Operations(find the value of the key, get the next N bytes)

A database is a collection of information organized to afford efficient retrieval. They solve four problems : Sharing data, Data enforcement model, Scale, Flexibility.

A definition of **Relational Database** is:

> Relational Database Management Systems were invented to let you use one set of data in multiple ways, including ways that are unforssen at the time the database is built and the 1st applications are written.

In relational database, everything is a table, every row in a table has the same columns, relationships are implicit: no pointers (shared Id). Key ideas are:

  - **Physical Data independence** a key idea of this family is that "activities of users should remain unaffected when the internal representation of data is changes" (always SQL queries). 
  - **Algebra of Tables**  select(select rows that satisfy some condition), project(ignore columns that you are not interested in)   join( for two tables, for every recore in the first table, find records in the second table).    *They are bad for multi tables queries*. 
  - **Algebraic Optimisation** the database will try to optimise the expressions you are giving to it
  - **indexes** Databaes are especially effective at "Needle in Haystack". indexes are easily built and automatically used.

### Relational algebra operators
 **Sets** *{a,b,c}, {a,d,e,f}*  and **bags** *{a,a,b,c}, {b,b,b,b}* are two notions we need to be familiar with. In normal relational algebra, set semantics will be assumed. In extended relational algebra, it might be bags. Main operators are :
 
  - **Union (u)** : `SELECT * FROM R1 UNION SELECT * FROM R2` 
  - **intersection (n)** : R1 - (R1-R2), things that are both in R1 and R2 ``
  - **difference (-)** : `SELECT * FROM R1 EXCEPT SELECT * FROM R2`
  - **selection (s or sigma)** : `SELECT * FROM R1 WHERE SALARY > 4000`
  - **projection (PI MAJ)** : eliminate columns
  - **cross product (x)** : each tuple in R1 with each tuple in R2
  - **equi-join or join(sablier horizontal)** : for every record in R1, find a recored in R2 that satisfy a condition `SELECT * FROM R1,R2 WHERE R1.A=R2.B` or `SELECT * FROM R1 JOIN R2 ON R1.A=R2.B`
  - **right outer-join (sablier horizontal - barre droite)** : for evey tuple inR1, find me a tuple in R2 where zip=zip and pad tuple of R1 without correpondance inR2 with null 
  - **theta-join** : a join that involves a predicate(condition) `SELECT * FROM R1, R2 WHERE abs(R1.time - R2.time) < 5`
  - **duplicate elimiation(d)** : `DISTINCT` example : `SELECT DISTINCT x.name, z.name FROM ..`
  - **grouping(g)** : `GROUP BY` example : `SELECT city,count(*) FROM sales GROUP BY city having sum(price)>100`
  - **sorting(s)** : `ORDER BY`

There is also a concept called **view**. A view is a query with a name and we can use it just like a real table! `CREATE VIEW viewName AS SELECT x.store FROM Purchase x WHERE ..`

## Map Reduce
Map Reduce come from a paper published by Google in 2004. It is an abstraction that is used like this : take an input as a bag of (inputkey, value) pairs and produce an output as a _bag_ of (outputkey, value) pairs.

 - **map** take (inputkey, value) and produce a _bag_ of (output-key, value).
 - **Reduce** take (output-key, bag of values) and produce a bag of output (output-key,values).

For example, to count word occurences across all documents, you could do like this: for each document make a pair (doc_id, text) then **map** will tranform this into a set of (word,1) . Then **reduce** will take (word,{1,1,1,1,..,1}) and transform it into (word,25). 
It is used over a file system. There is Google DFS which is proprietary and Hadoop's DFS : **HDFS** which is open source. In this, each file is partitioned into chunks.
There is quite a few implementations of MapReduce : Pig(Yahoo -> relational algebra over hadoop), HIVE(facebook), Impala(SQL over HDFS)    

Map reduce can fit into the following design Space like This:

|            |Shared memory   | data-parallel |
|------------|----------------|---------------|
|latency     | older database | No-SQL        |
| throughput |    HPC         | Analytics (MR)|

_MR = Map Reduce, Older database = SQL_

Key ideas are : **fault tolerance**, **no loading**, **direct programming** on "in situ" data, **single developer**.

## NoSQL
NoSQL are typically associated with building very large scalable web applications as opposed to analyzing data but they are becoming to be more used. Instead of ACID(Atomicity, Consistency, Isolation, Durability) like traditional database, they usually follows BASE(Basic Availability, Soft-state, Eventual consistency)
A major difference between Database and NoSQL is about consistency. For databases, "everyone MUST see the same thing, either old or new, no matter how long it takes" as for NoSQL it can operate over **Relaxed consistency guarantee** : "For large applications, we can't afford to wait that long, and maybe it doesn't matter anyway". This is what it means when a NoSQL transaction is **EC** (Eventual Consistency). 

Another difference is that conventional databases assume no partitioning whereas it's at the core of NoSQL. Moreover, NoSQL has the ability to horizontally scale "simple operation" throughput to many servers.

NoSQL are usually in one of these 3 store families:

1. **Document store** : nested values, extensible records (think XML or JSON)
2. **Extensible record** : families of attributes have a schema(like databases), but new attributes may be added
3. **Key-Value object** : a set of key-value pairs. No schema, no exposed nesting


Two pro NoSQL arguments are :

 - Performance : "I started with MySQL, but had a hard time scaling it out in a distributed environment"
 - Scalability : "My data doesn't conform to a rigid schema"

Some noSQL criticism are :

 - No ACID Equals no Interest (screw up mission-critical data is no-no-no)
 - Low-level Query Language is Death!
 - NoSQL means no standards (if you have 10k databaese, you need accepted standards)

### Pig
Language on top of Map-Reduce which is part of Hadoop. Quite like linear algebra. Reduce the number of lines a lot. so gain of programmer productivity.

### Spark
Quite a trend! More efficient than Hadoop systems like pig because it can do multiple operations in memory. It really on a concept called RDD : **Resilient Distributed Dataset**. An RDD is a Distributed Collection of Key-Value Pairs and it can be persistent in memory across tasks!.

Here is an example of  *word count* :

```
 val textfile = spark.textFile("hdfs://...")
 val counts = textfile.flapMap(line => line.split(" "))
                      .map(word => (word,1))
                      .reduceByKey(_ + _)
counts.saveAsTextFile("hdfs://...")
```

