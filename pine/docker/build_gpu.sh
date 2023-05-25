UBUNTU_VERSION="22.04"
CUDA_VERSION="11.2"

help() {
    echo "Usage: $0 [-cuda <cuda version. Default=11.2>] [-ubuntu <ubuntu version>. Default=22.04]";
    exit 1;
}

while getopt :cuda:ubuntu: versions; do
    case ${versions} in
        cuda) CUDA_VERSION=${OPTARG} ;;
        ubuntu) UBUNTU_VERSION=${OPTARG} ;;
        *) help ;;
    esac
done

docker build -t pine:latest -f DockerfileGPU --build-arg CUDA=$CUDA_VERSION --build-arg UBUNTU_VERSION=$UBUNTU_VERSION ..
