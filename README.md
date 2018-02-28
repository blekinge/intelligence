# intelligence
Experiments with Tensor-flow

Install Tensor-flow using method on https://www.tensorflow.org/install/install_java
Remember to download JNI packages for your platform

If you download the jni-files, and get warning:
implausibly old time stamp 1970-01-01 01:00:0

Your platform are not supported by the tensorflow software.
You will have link errors such as these,
if you 
run the command ldd jni/*.so


jni/libtensorflow_framework.so:
jni/libtensorflow_framework.so: /lib64/libc.so.6: version `GLIBC_2.14' not found (required by jni/libtensorflow_framework.so)
jni/libtensorflow_framework.so: /lib64/libc.so.6: version `GLIBC_2.17' not found (required by jni/libtensorflow_framework.so)
jni/libtensorflow_framework.so: /lib64/libc.so.6: version `GLIBC_2.16' not found (required by jni/libtensorflow_framework.so)
jni/libtensorflow_framework.so: /usr/lib64/libstdc++.so.6: version `GLIBCXX_3.4.21' not found (required by jni/libtensorflow_framework.so)
jni/libtensorflow_framework.so: /usr/lib64/libstdc++.so.6: version `GLIBCXX_3.4.14' not found (required by jni/libtensorflow_framework.so)
jni/libtensorflow_framework.so: /usr/lib64/libstdc++.so.6: version `GLIBCXX_3.4.18' not found (required by jni/libtensorflow_framework.so)
jni/libtensorflow_framework.so: /usr/lib64/libstdc++.so.6: version `CXXABI_1.3.5' not found (required by jni/libtensorflow_framework.so)
jni/libtensorflow_framework.so: /usr/lib64/libstdc++.so.6: version `GLIBCXX_3.4.20' not found (required by jni/libtensorflow_framework.so)
jni/libtensorflow_framework.so: /usr/lib64/libstdc++.so.6: version `GLIBCXX_3.4.19' not found (required by jni/libtensorflow_framework.so)
jni/libtensorflow_framework.so: /usr/lib64/libstdc++.so.6: version `GLIBCXX_3.4.17' not found (required by jni/libtensorflow_framework.so)
jni/libtensorflow_framework.so: /usr/lib64/libstdc++.so.6: version `CXXABI_1.3.8' not found (required by jni/libtensorflow_framework.so)
jni/libtensorflow_framework.so: /usr/lib64/libstdc++.so.6: version `CXXABI_1.3.7' not found (required by jni/libtensorflow_framework.so)
	linux-vdso.so.1 =>  (0x00007ffde1c5a000)
	libdl.so.2 => /lib64/libdl.so.2 (0x00007f1c005d9000)
	libm.so.6 => /lib64/libm.so.6 (0x00007f1c00354000)
	libpthread.so.0 => /lib64/libpthread.so.0 (0x00007f1c00137000)
	libstdc++.so.6 => /usr/lib64/libstdc++.so.6 (0x00007f1bffe31000)
	libgcc_s.so.1 => /lib64/libgcc_s.so.1 (0x00007f1bffc1a000)
	libc.so.6 => /lib64/libc.so.6 (0x00007f1bff886000)
	/lib64/ld-linux-x86-64.so.2 (0x0000003a8cc00000)
jni/libtensorflow_jni.so:
jni/libtensorflow_jni.so: /lib64/libc.so.6: version `GLIBC_2.16' not found (required by jni/libtensorflow_jni.so)
jni/libtensorflow_jni.so: /lib64/libc.so.6: version `GLIBC_2.14' not found (required by jni/libtensorflow_jni.so)
jni/libtensorflow_jni.so: /lib64/libc.so.6: version `GLIBC_2.15' not found (required by jni/libtensorflow_jni.so)
jni/libtensorflow_jni.so: /usr/lib64/libstdc++.so.6: version `GLIBCXX_3.4.20' not found (required by jni/libtensorflow_jni.so)
jni/libtensorflow_jni.so: /usr/lib64/libstdc++.so.6: version `CXXABI_1.3.5' not found (required by jni/libtensorflow_jni.so)
jni/libtensorflow_jni.so: /usr/lib64/libstdc++.so.6: version `GLIBCXX_3.4.18' not found (required by jni/libtensorflow_jni.so)
jni/libtensorflow_jni.so: /usr/lib64/libstdc++.so.6: version `GLIBCXX_3.4.21' not found (required by jni/libtensorflow_jni.so)
jni/libtensorflow_jni.so: /usr/lib64/libstdc++.so.6: version `CXXABI_1.3.7' not found (required by jni/libtensorflow_jni.so)
jni/libtensorflow_jni.so: /usr/lib64/libstdc++.so.6: version `GLIBCXX_3.4.17' not found (required by jni/libtensorflow_jni.so)
jni/libtensorflow_jni.so: /usr/lib64/libstdc++.so.6: version `CXXABI_1.3.8' not found (required by jni/libtensorflow_jni.so)
jni/libtensorflow_jni.so: /usr/lib64/libstdc++.so.6: version `GLIBCXX_3.4.19' not found (required by jni/libtensorflow_jni.so)
jni/libtensorflow_jni.so: /usr/lib64/libstdc++.so.6: version `GLIBCXX_3.4.14' not found (required by jni/libtensorflow_jni.so)
jni/libtensorflow_jni.so: /usr/lib64/libstdc++.so.6: version `GLIBCXX_3.4.15' not found (required by jni/libtensorflow_jni.so)
jni/libtensorflow_jni.so: /lib64/libm.so.6: version `GLIBC_2.23' not found (required by jni/libtensorflow_jni.so)
jni/libtensorflow_jni.so: /lib64/libc.so.6: version `GLIBC_2.14' not found (required by /home/svc/devel/intelligence/jni/libtensorflow_framework.so)
jni/libtensorflow_jni.so: /lib64/libc.so.6: version `GLIBC_2.17' not found (required by /home/svc/devel/intelligence/jni/libtensorflow_framework.so)
jni/libtensorflow_jni.so: /lib64/libc.so.6: version `GLIBC_2.16' not found (required by /home/svc/devel/intelligence/jni/libtensorflow_framework.so)
jni/libtensorflow_jni.so: /usr/lib64/libstdc++.so.6: version `GLIBCXX_3.4.21' not found (required by /home/svc/devel/intelligence/jni/libtensorflow_framework.so)
jni/libtensorflow_jni.so: /usr/lib64/libstdc++.so.6: version `GLIBCXX_3.4.14' not found (required by /home/svc/devel/intelligence/jni/libtensorflow_framework.so)
jni/libtensorflow_jni.so: /usr/lib64/libstdc++.so.6: version `GLIBCXX_3.4.18' not found (required by /home/svc/devel/intelligence/jni/libtensorflow_framework.so)
jni/libtensorflow_jni.so: /usr/lib64/libstdc++.so.6: version `CXXABI_1.3.5' not found (required by /home/svc/devel/intelligence/jni/libtensorflow_framework.so)
jni/libtensorflow_jni.so: /usr/lib64/libstdc++.so.6: version `GLIBCXX_3.4.20' not found (required by /home/svc/devel/intelligence/jni/libtensorflow_framework.so)
jni/libtensorflow_jni.so: /usr/lib64/libstdc++.so.6: version `GLIBCXX_3.4.19' not found (required by /home/svc/devel/intelligence/jni/libtensorflow_framework.so)
jni/libtensorflow_jni.so: /usr/lib64/libstdc++.so.6: version `GLIBCXX_3.4.17' not found (required by /home/svc/devel/intelligence/jni/libtensorflow_framework.so)
jni/libtensorflow_jni.so: /usr/lib64/libstdc++.so.6: version `CXXABI_1.3.8' not found (required by /home/svc/devel/intelligence/jni/libtensorflow_framework.so)
jni/libtensorflow_jni.so: /usr/lib64/libstdc++.so.6: version `CXXABI_1.3.7' not found (required by /home/svc/devel/intelligence/jni/libtensorflow_framework.so)
	linux-vdso.so.1 =>  (0x00007ffed01ce000)
	libtensorflow_framework.so => /home/svc/devel/intelligence/jni/libtensorflow_framework.so (0x00007f14119ae000)
	libdl.so.2 => /lib64/libdl.so.2 (0x00007f1411794000)
	libpthread.so.0 => /lib64/libpthread.so.0 (0x00007f1411577000)
	libm.so.6 => /lib64/libm.so.6 (0x00007f14112f3000)
	librt.so.1 => /lib64/librt.so.1 (0x00007f14110ea000)
	libstdc++.so.6 => /usr/lib64/libstdc++.so.6 (0x00007f1410de4000)
	libgcc_s.so.1 => /lib64/libgcc_s.so.1 (0x00007f1410bce000)
	libc.so.6 => /lib64/libc.so.6 (0x00007f1410839000)
	/lib64/ld-linux-x86-64.so.2 (0x0000003a8cc00000)




