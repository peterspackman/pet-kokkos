#!/bin/bash
# Add pair_style pet (package ML-PET, and pet/kk for the KOKKOS package) to a
# LAMMPS source tree, then build LAMMPS with -D PKG_ML-PET=ON.
#
#   patch_lammps.sh /path/to/lammps          copy the sources in
#   patch_lammps.sh /path/to/lammps --link   symlink them, for working on them
#
# Running it again updates the tree.
set -euo pipefail
lammps=$(cd "${1:?usage: patch_lammps.sh /path/to/lammps [--link]}" && pwd)
here=$(cd "$(dirname "$0")" && pwd)
pet=$(dirname "$here")
[ -f "$lammps/cmake/CMakeLists.txt" ] && [ -d "$lammps/src/KOKKOS" ] || { echo "not a LAMMPS source tree: $lammps" >&2; exit 1; }

link=0; [ "${2:-}" = "--link" ] && link=1
put() {  # put SRC DST: copy, or symlink with --link
  rm -rf "$2"
  if [ $link = 1 ]; then ln -s "$1" "$2"; else cp -r "$1" "$2"; fi
}

put "$here/ML-PET" "$lammps/src/ML-PET"
put "$here/KOKKOS/pair_pet_kokkos.h" "$lammps/src/KOKKOS/pair_pet_kokkos.h"
put "$here/KOKKOS/pair_pet_kokkos.cpp" "$lammps/src/KOKKOS/pair_pet_kokkos.cpp"
sed "s|@PET_KOKKOS_DIR@|$pet|" "$here/cmake/ML-PET.cmake" > "$lammps/cmake/Modules/Packages/ML-PET.cmake"

# Register the package: its name, and its module after the KOKKOS package's (so
# pet-kokkos can use LAMMPS's Kokkos).
cmake="$lammps/cmake/CMakeLists.txt"
if ! grep -q "ML-PET" "$cmake"; then
  sed -i -e 's/^\(\s*\)ML-PACE$/&\n\1ML-PET/' \
         -e 's/\(foreach(PKG_WITH_INCL CORESHELL .*KOKKOS\) /\1 ML-PET /' "$cmake"
fi
grep -q "^\s*ML-PET$" "$cmake" && grep -q "KOKKOS ML-PET" "$cmake" || { echo "could not register ML-PET in $cmake" >&2; exit 1; }
echo "ML-PET added to $lammps (pet-kokkos: $pet). Configure LAMMPS with -D PKG_ML-PET=ON."
