#include "pet/io.hpp"

#include <array>
#include <cctype>
#include <charconv>
#include <fstream>
#include <iomanip>
#include <ostream>
#include <sstream>
#include <stdexcept>

namespace pet {

namespace {

// Indexed by atomic number; 0 is "unknown".
constexpr const char* kSymbols[] = {
    "X",  "H",  "He", "Li", "Be", "B",  "C",  "N",  "O",  "F",  "Ne", "Na", "Mg", "Al", "Si",
    "P",  "S",  "Cl", "Ar", "K",  "Ca", "Sc", "Ti", "V",  "Cr", "Mn", "Fe", "Co", "Ni", "Cu",
    "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y",  "Zr", "Nb", "Mo", "Tc", "Ru",
    "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I",  "Xe", "Cs", "Ba", "La", "Ce", "Pr",
    "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W",
    "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac",
    "Th", "Pa", "U",  "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf",
    "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og"};
constexpr int kMaxZ = static_cast<int>(sizeof(kSymbols) / sizeof(kSymbols[0])) - 1;

// `key="quoted value"` or `key=bare_value` from an extxyz comment line.
std::string extract_key(const std::string& line, const std::string& key) {
  // Keys match case-insensitively: producers disagree on "Lattice" vs "lattice".
  auto lower = [](std::string x) {
    for (auto& c : x) c = (char) std::tolower((unsigned char) c);
    return x;
  };
  const std::string lower_line = lower(line), lower_key = lower(key);
  std::size_t p = 0;
  while ((p = lower_line.find(lower_key, p)) != std::string::npos) {
    // A whole key: at the start of the line or after a space, then '='.
    const bool at_start = (p == 0) || std::isspace((unsigned char) line[p - 1]);
    const std::size_t eq = p + key.size();
    if (at_start && eq < line.size() && line[eq] == '=') {
      std::size_t v = eq + 1;
      if (v < line.size() && (line[v] == '"' || line[v] == '\'')) {
        const char quote = line[v];
        const std::size_t end = line.find(quote, v + 1);
        if (end == std::string::npos) return {};
        return line.substr(v + 1, end - v - 1);
      }
      const std::size_t end = line.find_first_of(" \t", v);
      return line.substr(v, (end == std::string::npos ? line.size() : end) - v);
    }
    p = eq;
  }
  return {};
}

bool truthy(const std::string& s) {
  return !s.empty() && (s[0] == 'T' || s[0] == 't' || s[0] == '1');
}

}  // namespace

const char* element_symbol(int atomic_number) {
  if (atomic_number < 1 || atomic_number > kMaxZ) return kSymbols[0];
  return kSymbols[atomic_number];
}

int element_number(const std::string& symbol) {
  if (symbol.empty()) return 0;
  // Numeric species columns are legal extxyz too.
  if (std::isdigit((unsigned char) symbol[0])) {
    int z = 0;
    const auto r = std::from_chars(symbol.data(), symbol.data() + symbol.size(), z);
    return (r.ec == std::errc{}) ? z : 0;
  }
  // "he" and "HE" are "He".
  std::string s = symbol;
  s[0] = (char) std::toupper((unsigned char) s[0]);
  for (std::size_t i = 1; i < s.size(); ++i) s[i] = (char) std::tolower((unsigned char) s[i]);
  for (int z = 1; z <= kMaxZ; ++z)
    if (s == kSymbols[z]) return z;
  return 0;
}

std::vector<System> read_extxyz(std::istream& in) {
  std::vector<System> frames;
  std::string line;

  while (std::getline(in, line)) {
    if (line.find_first_not_of(" \t\r\n") == std::string::npos) continue;

    int n = 0;
    {
      std::istringstream ss(line);
      if (!(ss >> n) || n < 0)
        throw std::runtime_error("extxyz: expected an atom count, got '" + line + "'");
    }

    std::string comment;
    if (!std::getline(in, comment))
      throw std::runtime_error("extxyz: file ends after the atom count");

    System s;
    s.n_atoms = n;
    s.atomic_numbers.reserve(n);
    s.positions.reserve(static_cast<std::size_t>(n) * 3);

    // Lattice="ax ay az bx by bz cx cy cz": vectors as rows, like System::cell.
    const std::string lat = extract_key(comment, "Lattice");
    bool has_cell = false;
    if (!lat.empty()) {
      std::istringstream ls(lat);
      int i = 0;
      double v = 0;
      while (i < 9 && (ls >> v)) s.cell[i++] = v;
      if (i != 9)
        throw std::runtime_error("extxyz: Lattice needs 9 numbers, found " + std::to_string(i));
      has_cell = true;
    }

    // pbc="T T T"; absent, periodic if and only if there is a lattice (as ASE).
    const std::string pbc = extract_key(comment, "pbc");
    if (!pbc.empty()) {
      std::istringstream ps(pbc);
      std::string tok;
      for (int d = 0; d < 3 && (ps >> tok); ++d) s.pbc[d] = truthy(tok);
    } else {
      s.pbc = {has_cell, has_cell, has_cell};
    }

    for (int a = 0; a < n; ++a) {
      if (!std::getline(in, line))
        throw std::runtime_error("extxyz: file ends after " + std::to_string(a) + " of " +
                                 std::to_string(n) + " atoms");
      std::istringstream as(line);
      std::string species;
      double x = 0, y = 0, z = 0;
      if (!(as >> species >> x >> y >> z))
        throw std::runtime_error("extxyz: cannot parse atom line '" + line + "'");
      const int Z = element_number(species);
      if (Z == 0) throw std::runtime_error("extxyz: unknown species '" + species + "'");
      s.atomic_numbers.push_back(Z);
      s.positions.insert(s.positions.end(), {x, y, z});
    }
    frames.push_back(std::move(s));
  }
  return frames;
}

std::vector<System> read_extxyz(const std::string& path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("extxyz: cannot open '" + path + "'");
  return read_extxyz(f);
}

void write_extxyz(std::ostream& out, const System& s, const double* forces,
                  const double* energy) {
  out << s.n_atoms << '\n';

  const bool has_cell = s.pbc[0] || s.pbc[1] || s.pbc[2];
  out << std::setprecision(12);
  if (has_cell) {
    out << "Lattice=\"";
    for (int i = 0; i < 9; ++i) out << (i ? " " : "") << s.cell[i];
    out << "\" ";
  }
  out << "Properties=species:S:1:pos:R:3";
  if (forces) out << ":forces:R:3";
  out << " pbc=\"" << (s.pbc[0] ? 'T' : 'F') << ' ' << (s.pbc[1] ? 'T' : 'F') << ' '
      << (s.pbc[2] ? 'T' : 'F') << '"';
  if (energy) out << " energy=" << *energy;
  out << '\n';

  for (int a = 0; a < s.n_atoms; ++a) {
    out << element_symbol(s.atomic_numbers[a]);
    for (int c = 0; c < 3; ++c) out << ' ' << s.positions[static_cast<std::size_t>(a) * 3 + c];
    if (forces)
      for (int c = 0; c < 3; ++c) out << ' ' << forces[static_cast<std::size_t>(a) * 3 + c];
    out << '\n';
  }
}

}  // namespace pet
