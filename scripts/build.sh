script_dir="$(cd "$(dirname "$0")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"
cd "$repo_root"

build_type="${1:-debug}"

if [ "$build_type" = "release" ]; then
    cmake_build_type="Release"
    build_dir="build/release"
elif [ "$build_type" = "debug" ]; then
    cmake_build_type="Debug"
    build_dir="build/debug"
else
    echo "Usage: sh script/build.sh [debug|release]"
    exit 1
fi

cmake -S test -B "$build_dir" -DCMAKE_BUILD_TYPE="$cmake_build_type"
cmake --build "$build_dir" --target test_celeritas -j 32
