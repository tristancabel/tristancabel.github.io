---
layout: post
title: "dynamic libraries loading"
categories: debug
tags: debug libraries ldd nm readelf LD_DEBUG linux
---
When working with dynamic libraries we can sometimes wonder where does some dependencies come from! In this post, we will explore this issue and present the commands *ldd* and *readelf* to get more informations about dynamic libraries.
<!--more-->

# How are library loaded
To start, let's do a simple reminder of how dependencies are loaded on linux using **ld-linux**. It will look for libraries in the following order (from [ld-linux](http://linux.die.net/man/8/ld-linux) ):

 1. Using the environment variable *LD_LIBRARY_PATH*. Except if the executable is a set-user-ID/set-group-ID binary, in which case it is ignored.
 2. Using the directories specified in the *DT_RUNPATH* dynamic section attribute of the binary if present.
 3. From the cache file /etc/ld.so.cache, which contains a compiled list of candidate libraries previously found in the augmented library path. If, however, the binary was linked with the *-z nodeflib* linker option, libraries in the default library paths are skipped. Libraries installed in hardware capability directories (see below) are preferred to other libraries.
 4. In the default path /lib, and then /usr/lib. If the binary was linked with the *-z nodeflib* linker option, this step is skipped.


# Where does this dependency come from
We can now explore which dependencies will be loaded by our library with the command **ldd**:

{% highlight shell %}
tcabel@fantasy:/home/tcabel/De...e-geometry/build$ ldd lib64/libQVTKWidgetPlugin.so
	linux-vdso.so.1 =>  (0x00007ffc64b0c000)
	libQt5Widgets.so.5 => /lib64/libQt5Widgets.so.5 (0x00007f9f47f26000)
	libQt5Gui.so.5 => /lib64/libQt5Gui.so.5 (0x00007f9f4796f000)
	libQt5Core.so.5 => /lib64/libQt5Core.so.5 (0x00007f9f474a1000)
	libstdc++.so.6 => /lib64/libstdc++.so.6 (0x00007f9f47199000)
	libm.so.6 => /lib64/libm.so.6 (0x00007f9f46e96000)
	libc.so.6 => /lib64/libc.so.6 (0x00007f9f46ad5000)
	libgcc_s.so.1 => /lib64/libgcc_s.so.1 (0x00007f9f468bf000)
	libpthread.so.0 => /lib64/libpthread.so.0 (0x00007f9f466a2000)
	libgobject-2.0.so.0 => /lib64/libgobject-2.0.so.0 (0x00007f9f46452000)
	libglib-2.0.so.0 => /lib64/libglib-2.0.so.0 (0x00007f9f4611b000)
	libXext.so.6 => /lib64/libXext.so.6 (0x00007f9f45f08000)
	libX11.so.6 => /lib64/libX11.so.6 (0x00007f9f45bca000)
	libGL.so.1 => /usr/lib64/nvidia/libGL.so.1 (0x00007f9f4589a000)
	libpng15.so.15 => /lib64/libpng15.so.15 (0x00007f9f4566e000)
	libz.so.1 => /lib64/libz.so.1 (0x00007f9f45458000)
	libicui18n.so.50 => /lib64/libicui18n.so.50 (0x00007f9f45059000)
	libicuuc.so.50 => /lib64/libicuuc.so.50 (0x00007f9f44cdf000)
	libicudata.so.50 => /lib64/libicudata.so.50 (0x00007f9f4370b000)
	libpcre16.so.0 => /lib64/libpcre16.so.0 (0x00007f9f434b3000)
	libdl.so.2 => /lib64/libdl.so.2 (0x00007f9f432ae000)
	libgthread-2.0.so.0 => /lib64/libgthread-2.0.so.0 (0x00007f9f430ac000)
	librt.so.1 => /lib64/librt.so.1 (0x00007f9f42ea4000)
	/lib64/ld-linux-x86-64.so.2 (0x00007f9f487f4000)
	libffi.so.6 => /lib64/libffi.so.6 (0x00007f9f42c9b000)
	libxcb.so.1 => /lib64/libxcb.so.1 (0x00007f9f42a79000)
	libnvidia-tls.so.352.79 => /usr/lib64/nvidia/tls/libnvidia-tls.so.352.79 (0x00007f9f42875000)
	libnvidia-glcore.so.352.79 => /usr/lib64/nvidia/libnvidia-glcore.so.352.79 (0x00007f9f3fde1000)
	libXau.so.6 => /lib64/libXau.so.6 (0x00007f9f3fbdd000)
{% endhighlight %}

However, **ldd** print all the needed libraries, meaning libraries needed by our library and dependencies of libraries loaded!
In the example above, let's say the dependency to *libX11* is annoying me and I want to know where does it come from.

The first thing I can do is checking it doesn't come directly from my lib printing only it's direct dependencies thanks to **readelf**:

{% highlight shell %}
tcabel@fantasy:/home/tcabel/De...e-geometry/build$readelf -d lib64/libQVTKWidgetPlugin.so

Dynamic section at offset 0xad50 contains 31 entries:
  Tag        Type                         Name/Value
 0x0000000000000001 (NEEDED)             Shared library: [libQt5Widgets.so.5]
 0x0000000000000001 (NEEDED)             Shared library: [libQt5Gui.so.5]
 0x0000000000000001 (NEEDED)             Shared library: [libQt5Core.so.5]
 0x0000000000000001 (NEEDED)             Shared library: [libstdc++.so.6]
 0x0000000000000001 (NEEDED)             Shared library: [libm.so.6]
 0x0000000000000001 (NEEDED)             Shared library: [libc.so.6]
 0x0000000000000001 (NEEDED)             Shared library: [libgcc_s.so.1]
 0x000000000000000e (SONAME)             Library soname: [libQVTKWidgetPlugin.so]
 0x000000000000000f (RPATH)              Library rpath: [/user/tcabel/home/Devel/dtk/build/lib64:/home/tcabel/lib/QT-5.4/5.4/gcc_64/lib:]
{% endhighlight %}

As it name implies readelf reads ELF information (which is the format of dynamic libraries on Linux) and as we can see there is no direct dependencies to *libX11*  in my lib :) .

But the question remain : where is libX11 coming from? We can find the culprit using the environment variable **LD_DEBUG**:

{% highlight shell %}
tcabel@fantasy:/home/tcabel/De...e-geometry/build$LD_DEBUG=files ldd  lib64/libQVTKWidgetPlugin.so

13861:	file=libGL.so.1 [0];  needed by /user/tcabel/home/lib/QT-5.4/5.4/gcc_64/lib/libQt5Gui.so [0]
     13861:	file=libGL.so.1 [0];  generating link map
     13861:	  dynamic: 0x00007f6326978038  base: 0x00007f63266eb000   size: 0x000000000028eef8
     13861:	    entry: 0x00007f6326717710  phdr: 0x00007f63266eb040  phnum:                  4
[...]
13861:	file=libGLX.so.0 [0];  needed by /lib64/libGL.so.1 [0]
     13861:	file=libGLX.so.0 [0];  generating link map
     13861:	  dynamic: 0x00007f632472e7c8  base: 0x00007f632451f000   size: 0x000000000022ffa0
     13861:	    entry: 0x00007f6324522770  phdr: 0x00007f632451f040  phnum:                  4
[...]
13861:	file=libGLX.so.0 [0];  needed by /lib64/libGL.so.1 [0]
     13861:	file=libGLX.so.0 [0];  generating link map
     13861:	  dynamic: 0x00007f632472e7c8  base: 0x00007f632451f000   size: 0x000000000022ffa0
     13861:	    entry: 0x00007f6324522770  phdr: 0x00007f632451f040  phnum:
[...]
13861:	file=libX11.so.6 [0];  needed by /lib64/libGLX.so.0 [0]
     13861:	file=libX11.so.6 [0];  generating link map
     13861:	  dynamic: 0x00007f6322ba7630  base: 0x00007f632286d000   size: 0x000000000033f8f8
     13861:	    entry: 0x00007f632288afe0  phdr: 0x00007f632286d040  phnum:
{% endhighlight %}

Then we can solve the puzzle *libQVTKWidgetPlugin->libQt5Gui->libGL->libGLX->libX11* !!
