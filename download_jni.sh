#TF_TYPE="cpu" # Default processor is CPU. If you want GPU, set to "gpu"
TF_TYPE="cpu"
VERSION=1.6.0
PLATFORM=x86_64
OS=$(uname -s | tr '[:upper:]' '[:lower:]')
echo "Trying to download $TF_TYPE based library for Tensorflow v. $VERSION for OS=$OS and platform=$PLATFORM"

 mkdir -p ./jni
 curl -L \
   "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow_jni-${TF_TYPE}-${OS}-${PLATFORM}-${VERSION}.tar.gz" |
   tar -xz -C ./jni

