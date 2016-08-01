---
layout: post
title: "scala tricks"
categories: programmation
tags: scala dev programmation
---

Scala tricks from different sources
<!--more-->


## serialization


## Side Effects

“Side effect” is an essential concept that should be reviewed before enumerating the actual transformations.

Basically, side effect is any action besides returning a value, that is observable outside of a function or expression scope, like:

    - input/output operation,
    - variable modification (that is accessible outside of the scope),
    - object state modification (that is observable outside of the scope),
    - throwing an exception (that is not caught within the scope).

When a function or expression contain any of these actions, they are said to have side effects, otherwise they are said to be **pure**.

Why side effects are such a big deal? Because in the presence of side effects, the order of evaluation matters. For example, here are two “pure” expressions (assigned to the corresponding values):

```
val x = 1 + 2
val y = 2 + 3
```

Because they contain no side effects (i. e. effects that are observable outside of the expressions), we may actually evaluate those expressions in any order — x then y, or y then x — it doesn’t disrupt evaluation correctness (we may even cache the result values, if we want to). Now let’s consider the following modification:

```
val x = { print("foo"); 1 + 2 }
val y = { print("bar"); 2 + 3 }
```

That’s another story — we cannot reverse the order of evaluation, because, that way, “barfoo” instead of “foobar” will be printed to console (and that is not what we expect).

So, the presence of side effects reduces the number of possible transformations (including simplifications and optimizations) that we may apply to code.


## Sequences

 - Create empty collections explicitly `Seq.empty[T]`. Some immutable collection classes provide singleton “empty” implementations, however not all of the factory methods check length of the created collections. Thus, by making collection emptiness apparent at compile time, we could save either heap space (by reusing empty collection instances) or CPU cycles (otherwise wasted on runtime length checks).
 - Prefer `length` to `size` for arrays. While size and length are basically synonyms, in Scala 2.11 Array.size calls are still implemented via implicit conversion, so that intermediate wrapper objects are created for every method call. Unless you enable escape analysis in JVM , those temporary objects will burden GC and can potentially degrade code performance (especially, within loops).
 - Don’t compute length for emptiness check:  `NOT seq.length > 0  YES seq.nonEmpty`
 - Don’t compute full length for length matching. Because length calculation might be “expensive” computation for some collection classes, we can reduce comparison time from O(length) to O(length min n) for decedents of LinearSeq (which might be hidden behind Seq-typed values). Besides, such approach is indispensable when we’re dealing with infinite streams.


```
// Before
seq.length > n
seq.length < n
seq.length == n
seq.length != n

// After
seq.lengthCompare(n) > 0
seq.lengthCompare(n) < 0
seq.lengthCompare(n) == 0
seq.lengthCompare(n) != 0
```

 - Equality, son’t rely on == to compare array contents use : `array1.sameElements(array2)`
 - Indexing  use `head` and `last` for first and last element, not `elem(0)`
 - Don’t check index bounds explicitly, the second expression is semantically equivalent, yet more concise.

```
// Before
if (i < seq.length) Some(seq(i)) else None

// After
seq.lift(i)
```

 - Don’t resort to filtering to check existence. The call to filter creates an intermediate collection which takes heap space and loads GC. Besides, the former expressions find all occurrences, when only the first one is needed (which might slowdown code, depending on likely collection contents). The potential performance gain is less significant for lazy collections (like Stream and, especially, Iterator). The predicate p must be pure.

```
// Before
seq.filter(p).nonEmpty
seq.filter(p).isEmpty

// After
seq.exists(p)
!seq.exists(p)
```




# References

  - [pavelfatin](https://pavelfatin.com/scala-collections-tips-and-tricks/)

