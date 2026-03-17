script_dir="$(cd "$(dirname "$0")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"
cd "$repo_root"

cmake -S test -B build/release -DCMAKE_BUILD_TYPE=Release
cmake --build build/release --target test_celeritas -j 32
cd build/release
ncu -f --set full --import-source yes -k row_rmsnorm_f32 -o report ./test_celeritas
