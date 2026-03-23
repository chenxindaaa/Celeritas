script_dir="$(cd "$(dirname "$0")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"
cd "$repo_root"

build_type="${1:-debug}"

if [ "$build_type" = "release" ]; then
    cmake_build_type="Release"
    build_dir="build/release"
    extra_cmake_args=""
elif [ "$build_type" = "debug" ]; then
    cmake_build_type="Debug"
    build_dir="build/debug"
    extra_cmake_args=""
elif [ "$build_type" = "asan" ]; then
    cmake_build_type="Debug"
    build_dir="build/asan"
    extra_cmake_args="-DENABLE_ASAN=ON"
else
    echo "Usage: sh scripts/build.sh [debug|release|asan]"
    exit 1
fi

cmake -S test -B "$build_dir" -DCMAKE_BUILD_TYPE="$cmake_build_type" $extra_cmake_args
cmake --build "$build_dir" --target test_celeritas -j 32
