// Extended-XYZ reading and writing. No model needed.
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <sstream>

#include "pet/io.hpp"

using Catch::Matchers::WithinAbs;

TEST_CASE("element symbols round-trip", "[io]") {
  CHECK(pet::element_number("H") == 1);
  CHECK(pet::element_number("C") == 6);
  CHECK(pet::element_number("Og") == 118);
  CHECK(std::string(pet::element_symbol(1)) == "H");
  CHECK(std::string(pet::element_symbol(118)) == "Og");
  // Case is normalized, because files in the wild are not consistent about it.
  CHECK(pet::element_number("he") == 2);
  CHECK(pet::element_number("HE") == 2);
  // A numeric species column is legal extxyz.
  CHECK(pet::element_number("26") == 26);
  // And an unknown symbol is reported rather than guessed at.
  CHECK(pet::element_number("Zz") == 0);
  CHECK(pet::element_number("") == 0);
  // Out-of-range atomic numbers do not walk off the table.
  CHECK(std::string(pet::element_symbol(0)) == "X");
  CHECK(std::string(pet::element_symbol(9999)) == "X");
  CHECK(std::string(pet::element_symbol(-1)) == "X");
}

TEST_CASE("a non-periodic frame reads back", "[io]") {
  std::istringstream in(
      "3\n"
      "Properties=species:S:1:pos:R:3 pbc=\"F F F\"\n"
      "O 0.0 0.0 0.11926\n"
      "H 0.0 0.76323 -0.47704\n"
      "H 0.0 -0.76323 -0.47704\n");
  const auto frames = pet::read_extxyz(in);
  REQUIRE(frames.size() == 1);
  const pet::System& s = frames[0];
  CHECK(s.n_atoms == 3);
  CHECK(s.atomic_numbers == std::vector<int>{8, 1, 1});
  CHECK_THAT(s.positions[2], WithinAbs(0.11926, 1e-12));
  CHECK_THAT(s.positions[4], WithinAbs(0.76323, 1e-12));
  CHECK_FALSE(s.pbc[0]);
  CHECK_FALSE(s.pbc[1]);
  CHECK_FALSE(s.pbc[2]);
}

TEST_CASE("a Lattice= entry is read as rows and implies periodicity", "[io]") {
  // Non-cubic, so a transposed cell would show.
  std::istringstream in(
      "1\n"
      "Lattice=\"1 2 3 4 5 6 7 8 9\" Properties=species:S:1:pos:R:3\n"
      "Si 0.1 0.2 0.3\n");
  const auto frames = pet::read_extxyz(in);
  REQUIRE(frames.size() == 1);
  const pet::System& s = frames[0];
  for (int i = 0; i < 9; ++i) CHECK_THAT(s.cell[i], WithinAbs(i + 1, 1e-12));
  // No pbc= entry, but a lattice is present -- ASE's writer implies periodic.
  CHECK(s.pbc[0]);
  CHECK(s.pbc[1]);
  CHECK(s.pbc[2]);
}

TEST_CASE("an explicit pbc= entry overrides the lattice default", "[io]") {
  std::istringstream in(
      "1\n"
      "Lattice=\"10 0 0 0 10 0 0 0 10\" pbc=\"T T F\" Properties=species:S:1:pos:R:3\n"
      "C 0 0 0\n");
  const auto frames = pet::read_extxyz(in);
  REQUIRE(frames.size() == 1);
  CHECK(frames[0].pbc[0]);
  CHECK(frames[0].pbc[1]);
  CHECK_FALSE(frames[0].pbc[2]);
}

TEST_CASE("multiple frames and extra columns", "[io]") {
  // Further columns (forces here) are ignored, not rejected.
  std::istringstream in(
      "2\n"
      "Properties=species:S:1:pos:R:3:forces:R:3\n"
      "H 0 0 0  0.1 0.2 0.3\n"
      "H 0 0 1 -0.1 -0.2 -0.3\n"
      "\n"
      "1\n"
      "Properties=species:S:1:pos:R:3\n"
      "He 1 1 1\n");
  const auto frames = pet::read_extxyz(in);
  REQUIRE(frames.size() == 2);
  CHECK(frames[0].n_atoms == 2);
  CHECK(frames[1].n_atoms == 1);
  CHECK(frames[1].atomic_numbers[0] == 2);
}

TEST_CASE("malformed files are rejected with a message, not silently truncated", "[io]") {
  SECTION("fewer atom lines than the count promises") {
    std::istringstream in("3\ncomment\nH 0 0 0\n");
    CHECK_THROWS(pet::read_extxyz(in));
  }
  SECTION("an unparseable atom line") {
    std::istringstream in("1\ncomment\nH 0 0\n");
    CHECK_THROWS(pet::read_extxyz(in));
  }
  SECTION("an unknown species") {
    std::istringstream in("1\ncomment\nZz 0 0 0\n");
    CHECK_THROWS(pet::read_extxyz(in));
  }
  SECTION("a Lattice with the wrong number of entries") {
    std::istringstream in("1\nLattice=\"1 2 3\"\nH 0 0 0\n");
    CHECK_THROWS(pet::read_extxyz(in));
  }
  SECTION("a missing comment line") {
    std::istringstream in("1\n");
    CHECK_THROWS(pet::read_extxyz(in));
  }
}

TEST_CASE("writing then reading gives back the same structure", "[io]") {
  pet::System s;
  s.n_atoms = 2;
  s.atomic_numbers = {8, 1};
  s.positions = {0.0, 0.5, 1.25, 2.0, -1.5, 0.125};
  s.cell = {5.1, 0.0, 0.0, 0.0, 5.2, 0.0, 0.1, 0.0, 5.3};
  s.pbc = {true, true, true};
  const double forces[6] = {0.1, -0.2, 0.3, -0.1, 0.2, -0.3};
  const double energy = -12.5;

  std::ostringstream out;
  pet::write_extxyz(out, s, forces, &energy);

  std::istringstream in(out.str());
  const auto back = pet::read_extxyz(in);
  REQUIRE(back.size() == 1);
  CHECK(back[0].n_atoms == s.n_atoms);
  CHECK(back[0].atomic_numbers == s.atomic_numbers);
  for (std::size_t i = 0; i < s.positions.size(); ++i)
    CHECK_THAT(back[0].positions[i], WithinAbs(s.positions[i], 1e-10));
  for (int i = 0; i < 9; ++i) CHECK_THAT(back[0].cell[i], WithinAbs(s.cell[i], 1e-10));
  CHECK(back[0].pbc == s.pbc);
}
