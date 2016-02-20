---
layout: post
title: "NaN debugging"
categories: debug
tags: debug NaN
---
I recently came accross a bug in a software I am contributing to causing some NaN to appear. In this software, I want to solve a system **Ax=b** using a sparse linear solver( *Hypre* ) but it failed to converge showing this message:

{% highlight bash %}
ERROR detected by Hypre ...  BEGIN
ERROR -- hypre_PCGSolve: INFs and/or NaNs detected in input.
User probably placed non-numerics in supplied b.
Returning error flag += 101.  Program not terminated.
ERROR detected by Hypre ...  END
Hypre PCG finish in 0 iterations with a final res norm of 0
{% endhighlight %}

I was about to start a long and complex debugging to find where these NaNs came from when I discovered this question [stakoverflow fortran NaN](http://stackoverflow.com/questions/5636580/force-gfortran-to-stop-program-at-first-nan) talking about a pretty useful C function define in **fenv.h**:
{% highlight C %} feenableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW); {% endhighlight %}  

Inserting this in my main allowed me to get this error :

{% highlight bash %}
 *** Process received signal ***
 Signal: Floating point exception (8)
 Signal code: Floating point divide-by-zero (3)
 Failing at address: 0x7fd8d9800bf5
 [ 0] /lib64/libc.so.6[0x3ad9834960]
 [ 1] /home/tcabel/Devel/tracesinria/build/lib/libtraces_para.so(comp_bk_24__+0xd9d)[0x7fd8d9800bf5]
 [ 2] /home/tcabel/Devel/tracesinria/build/lib/libtraces_para.so(__mod_inv_bk_MOD_inv_bk_all+0x552)[0x7fd8d97fcd40]
 [ 3] /home/tcabel/Devel/tracesinria/build/lib/libtraces_para.so(__mod_inv_bk_MOD_inv_bk+0x5b1)[0x7fd8d97fda8b]
 [ 4] /home/tcabel/Devel/tracesinria/build/lib/libtraces_para.so(__mod_comp_matrix_MOD_comp_matrix+0xe8e)[0x7fd8d9826069]
 [ 5] /home/tcabel/Devel/tracesinria/build/lib/libtraces_para.so(__mod_comp_transport_MOD_assemble_transport+0x1cd2)[0x7fd8d991666d]
 [ 6] /home/tcabel/Devel/tracesinria/build/lib/libtraces_para.so(__mod_traces_assemble_MOD_traces_assemble_transport+0x250)[0x7fd8d98484b9]
 [ 7] /home/tcabel/Devel/tracesinria/build/lib/libtraces_para.so(tracesAssembleNum3sis+0x46d)[0x7fd8d98480b3]
 [ 8] /home/tcabel/Devel/num3sis-plugins-private/build/plugins/libnumTracesAssemblyC4Plugin.so(_ZN19numTracesAssemblyC46updateEv+0x15e)[0x7fd8d9e0f720]
 [ 9] /home/tcabel/Devel/num3sis/build/lib64/libnumTraces.so(_ZN13numTracesExec3runEv+0xe29)[0x7fd90647ebfd]
 [10] /home/tcabel/Devel/dtk-plugins-distributed/build/plugins/libmpi3DistributedCommunicator.so(_ZN27mpi3DistributedCommunicator4execEP9QRunnable+0x4c)[0x7fd8dde7e9cc]
 [11] /home/tcabel/Devel/num3sis/build/bin/numTracesApp(main+0xf7f)[0x40e025]
{% endhighlight %}

From this, it is quite straitforward to find the faulty function *comp_bk_24* and the operation *divide by 0* creating the NaN. 

In case you are using Fortran, you can put some compile options to get the same functionnality : 

- **gfortran** : -ffpe-trap=invalid,zero,overflow,underflow
- **ifort**    :  -fpe0 

