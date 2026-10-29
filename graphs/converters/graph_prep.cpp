// graph_prep.cpp
// One-time preparation of a benchmark graph. Reads a DIMACS 9 (.gr text),
// Matrix Market (.mtx) or SNAP-style edge list, cleans it, and writes every
// file the benchmark frameworks need into one directory:
//
//   G.gr        Galois binary .gr v1, u32 weights   (Galois, LonestarGPU, MBQ,
//                                                     MLMQ/ADDS, ece496)
//   G.tgr       transpose of G.gr       (only if G is directed; Galois PR-pull)
//   G.sgr       symmetrized G.gr        (only if G is directed; MIS)
//   G.mtx       weighted Matrix Market  (Gunrock; "symmetric" storage when G
//                                         is undirected)
//   G.sym.mtx   symmetrized G.mtx       (only if G is directed)
//   G.sym.ag    PBBS AdjacencyGraph text of the symmetric graph (MBQ MIS)
//   G.bin       MBQ/Swarm A* binary     (only with --coords)
//   sources.txt, pairs.txt, reference.txt, info.txt
//
// Cleaning: self loops are dropped, duplicate edges keep the smallest weight,
// and each node's edges are sorted by destination. Node ids are kept as in the
// input (converted to 0-based).
//
// Weights (--weights):
//   input      the weight column of the input file (DIMACS)
//   random     integer in [wmin, wmax] from a hash of {min(u,v), max(u,v)} and
//              the seed, so (u,v) and (v,u) always match
//   haversine  great-circle length in metres from --coords, rounded, >= 1
//
// Coordinates (--coords-format):
//   dimacs-co  "v id x y" lines, x = lon*1e6, y = lat*1e6 (DIMACS 9 .co)
//   xyz        "x y z" line per node in node order, x = lon*1e5, y = lat*1e5
//              (DIMACS 10 .xyz)
//
// Inputs ending in .gz / .bz2 / .zip are decompressed through gzip, bzip2 or
// unzip (the *.mtx member). The whole graph is held in memory, so this is
// meant for graphs up to a few hundred million edges.
//
// Build: g++ -O3 -std=c++17 -o graph_prep graph_prep.cpp
// Usage: ./graph_prep --name G --out DIR --format dimacs|mtx|snap [options] <input>

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <numeric>
#include <queue>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "graph_prep_common.h"

static void usage(const char* prog) {
  std::fprintf(
      stderr,
      "Usage: %s --name G --out DIR --format dimacs|mtx|snap [options] "
      "<input>\n"
      "  --weights input|random|haversine  (default: input for dimacs, else "
      "random)\n"
      "  --wmin N --wmax N        random weight range (default 1 100)\n"
      "  --seed N                 seed for weights, sources, pairs (default "
      "1)\n"
      "  --one-based              snap ids start at 1\n"
      "  --coords FILE --coords-format dimacs-co|xyz\n"
      "  --bin-unit-m X           metres per weight unit for G.bin (default "
      "1)\n"
      "  --num-sources N          random sources besides the main one "
      "(default 10)\n"
      "  --num-pairs N            random + N farthest pairs (default 10)\n",
      prog);
  std::exit(1);
}

constexpr double kEarthRadiusM = 6371000.0; // same radius as MBQ astar
constexpr double kPi = 3.14159265358979323846;

// Great-circle central angle in radians (correct haversine).
static double central_angle(double lat1, double lon1, double lat2,
                            double lon2) {
  const double s1 = std::sin((lat2 - lat1) / 2), s2 = std::sin((lon2 - lon1) / 2);
  const double a = s1 * s1 + std::cos(lat1) * std::cos(lat2) * s2 * s2;
  return 2 * std::atan2(std::sqrt(a), std::sqrt(1 - a));
}

struct Edge {
  std::uint32_t u, v, w;
};

static bool edge_less(const Edge& x, const Edge& y) {
  return x.u != y.u ? x.u < y.u : (x.v != y.v ? x.v < y.v : x.w < y.w);
}

// Sorts, removes self loops and keeps the lightest copy of duplicate edges.
static void normalize(std::vector<Edge>& e) {
  e.erase(std::remove_if(e.begin(), e.end(),
                         [](const Edge& x) { return x.u == x.v; }),
          e.end());
  std::sort(e.begin(), e.end(), edge_less);
  e.erase(std::unique(e.begin(), e.end(),
                      [](const Edge& x, const Edge& y) {
                        return x.u == y.u && x.v == y.v;
                      }),
          e.end());
  e.shrink_to_fit();
}

struct CSR {
  std::uint64_t n = 0;
  std::vector<std::uint64_t> off; // n + 1
  std::vector<std::uint32_t> dst, w;
};

// Edges must be sorted by (u, v).
static CSR build_csr(std::uint64_t n, const std::vector<Edge>& e) {
  CSR g;
  g.n = n;
  g.off.assign(n + 1, 0);
  g.dst.resize(e.size());
  g.w.resize(e.size());
  for (std::size_t i = 0; i < e.size(); ++i) {
    ++g.off[e[i].u + 1];
    g.dst[i] = e[i].v;
    g.w[i] = e[i].w;
  }
  for (std::uint64_t i = 0; i < n; ++i)
    g.off[i + 1] += g.off[i];
  return g;
}

static void write_gr(const std::string& path, const CSR& g) {
  Writer out(path);
  const std::uint64_t m = g.dst.size();
  const std::uint64_t header[4] = {1, sizeof(std::uint32_t), g.n, m};
  out.raw(header, sizeof(header));
  out.raw(g.off.data() + 1, 8 * g.n); // end offsets
  out.raw(g.dst.data(), 4 * m);
  if (m % 2)
    out.bin(std::uint32_t{0});
  out.raw(g.w.data(), 4 * m);
}

// Reads a .gr back and checks it is identical to g.
static void verify_gr(const std::string& path, const CSR& g) {
  std::FILE* f = std::fopen(path.c_str(), "rb");
  if (!f)
    die("cannot reopen " + path);
  std::uint64_t h[4];
  bool ok = std::fread(h, 8, 4, f) == 4 && h[0] == 1 && h[1] == 4 &&
            h[2] == g.n && h[3] == g.dst.size();
  const std::uint64_t m = g.dst.size();
  std::vector<std::uint64_t> off(g.n);
  std::vector<std::uint32_t> d(m + (m & 1)), w(m);
  ok = ok && std::fread(off.data(), 8, g.n, f) == g.n &&
       std::fread(d.data(), 4, d.size(), f) == d.size() &&
       std::fread(w.data(), 4, m, f) == m && std::fgetc(f) == EOF;
  std::fclose(f);
  ok = ok && std::equal(off.begin(), off.end(), g.off.begin() + 1) &&
       std::equal(g.dst.begin(), g.dst.end(), d.begin()) && w == g.w;
  if (!ok)
    die("verification failed for " + path);
}

static bool is_symmetric(const CSR& g, bool& weights_match) {
  weights_match = true;
  for (std::uint64_t u = 0; u < g.n; ++u) {
    for (std::uint64_t i = g.off[u]; i < g.off[u + 1]; ++i) {
      const std::uint32_t v = g.dst[i];
      const auto b = g.dst.begin() + g.off[v], e = g.dst.begin() + g.off[v + 1];
      const auto it = std::lower_bound(b, e, static_cast<std::uint32_t>(u));
      if (it == e || *it != u)
        return false;
      if (g.w[it - g.dst.begin()] != g.w[i])
        weights_match = false;
    }
  }
  return true;
}

// Matrix Market with integer weights. symmetric=true writes only u > v
// entries ("symmetric" storage); the caller guarantees g is symmetric with
// matching weights.
static void write_mtx(const std::string& path, const CSR& g, bool symmetric) {
  Writer out(path);
  std::uint64_t nnz = 0;
  for (std::uint64_t u = 0; u < g.n; ++u)
    for (std::uint64_t i = g.off[u]; i < g.off[u + 1]; ++i)
      if (!symmetric || u > g.dst[i])
        ++nnz;
  out.str(symmetric ? "%%MatrixMarket matrix coordinate integer symmetric\n"
                    : "%%MatrixMarket matrix coordinate integer general\n");
  out.num(g.n);
  out.ch(' ');
  out.num(g.n);
  out.ch(' ');
  out.num(nnz);
  out.ch('\n');
  for (std::uint64_t u = 0; u < g.n; ++u)
    for (std::uint64_t i = g.off[u]; i < g.off[u + 1]; ++i)
      if (!symmetric || u > g.dst[i]) {
        out.num(u + 1);
        out.ch(' ');
        out.num(std::uint64_t{g.dst[i]} + 1);
        out.ch(' ');
        out.num(g.w[i]);
        out.ch('\n');
      }
}

static void write_ag(const std::string& path, const CSR& g) {
  Writer out(path);
  out.str("AdjacencyGraph\n");
  out.num(g.n);
  out.ch('\n');
  out.num(g.dst.size());
  out.ch('\n');
  for (std::uint64_t u = 0; u < g.n; ++u) {
    out.num(g.off[u]);
    out.ch('\n');
  }
  for (std::uint32_t v : g.dst) {
    out.num(v);
    out.ch('\n');
  }
}

// MBQ/Swarm A* format (mbq/astar/util.h): u32 magic, u32 n, then per node
// f64 lat, f64 lon (radians), u32 deg, u32 nbr[deg], f64 dist[deg] (radians
// of arc on a sphere of radius 6371 km). Edge length is the road length
// (weight * unit_m) but never less than the straight-line distance plus 1 cm,
// so a correct haversine heuristic stays admissible and consistent.
static std::uint64_t write_bin(const std::string& path, const CSR& g,
                               const std::vector<double>& lat,
                               const std::vector<double>& lon, double unit_m) {
  Writer out(path);
  out.bin(std::uint32_t{0x150842A7});
  out.bin(static_cast<std::uint32_t>(g.n));
  std::uint64_t clamped = 0;
  for (std::uint64_t u = 0; u < g.n; ++u) {
    out.bin(lat[u]);
    out.bin(lon[u]);
    const std::uint64_t b = g.off[u], e = g.off[u + 1];
    out.bin(static_cast<std::uint32_t>(e - b));
    out.raw(g.dst.data() + b, 4 * (e - b));
    for (std::uint64_t i = b; i < e; ++i) {
      const std::uint32_t v = g.dst[i];
      const double straight_m =
          central_angle(lat[u], lon[u], lat[v], lon[v]) * kEarthRadiusM + 0.01;
      double len_m = g.w[i] * unit_m;
      if (len_m < straight_m) {
        len_m = straight_m;
        ++clamped;
      }
      out.bin(len_m / kEarthRadiusM);
    }
  }
  return clamped;
}

static std::vector<std::uint32_t> bfs(const CSR& g, std::uint32_t s) {
  std::vector<std::uint32_t> level(g.n, UINT32_MAX);
  std::vector<std::uint32_t> q;
  q.reserve(1 << 20);
  level[s] = 0;
  q.push_back(s);
  for (std::size_t h = 0; h < q.size(); ++h) {
    const std::uint32_t u = q[h];
    for (std::uint64_t i = g.off[u]; i < g.off[u + 1]; ++i)
      if (level[g.dst[i]] == UINT32_MAX) {
        level[g.dst[i]] = level[u] + 1;
        q.push_back(g.dst[i]);
      }
  }
  return level;
}

static std::vector<std::uint64_t> dijkstra(const CSR& g, std::uint32_t s) {
  std::vector<std::uint64_t> dist(g.n, UINT64_MAX);
  using Item = std::pair<std::uint64_t, std::uint32_t>;
  std::priority_queue<Item, std::vector<Item>, std::greater<Item>> pq;
  dist[s] = 0;
  pq.push({0, s});
  while (!pq.empty()) {
    const auto [d, u] = pq.top();
    pq.pop();
    if (d != dist[u])
      continue;
    for (std::uint64_t i = g.off[u]; i < g.off[u + 1]; ++i) {
      const std::uint64_t nd = d + g.w[i];
      if (nd < dist[g.dst[i]]) {
        dist[g.dst[i]] = nd;
        pq.push({nd, g.dst[i]});
      }
    }
  }
  return dist;
}

struct Summary {
  std::uint64_t reached = 0, max = 0, sum = 0;
};

template <typename T> static Summary summarize(const std::vector<T>& d) {
  Summary s;
  for (T x : d)
    if (x != std::numeric_limits<T>::max()) {
      ++s.reached;
      s.max = std::max<std::uint64_t>(s.max, x);
      s.sum += x;
    }
  return s;
}

// Largest weakly connected component of g (sym is its symmetrization).
static std::vector<char> largest_wcc(const CSR& sym) {
  std::vector<std::uint32_t> comp(sym.n, UINT32_MAX), q;
  std::vector<std::uint64_t> size;
  for (std::uint64_t s = 0; s < sym.n; ++s) {
    if (comp[s] != UINT32_MAX)
      continue;
    const std::uint32_t c = size.size();
    size.push_back(0);
    q.assign(1, s);
    comp[s] = c;
    for (std::size_t h = 0; h < q.size(); ++h) {
      const std::uint32_t u = q[h];
      ++size[c];
      for (std::uint64_t i = sym.off[u]; i < sym.off[u + 1]; ++i)
        if (comp[sym.dst[i]] == UINT32_MAX) {
          comp[sym.dst[i]] = c;
          q.push_back(sym.dst[i]);
        }
    }
  }
  const std::uint32_t best =
      std::max_element(size.begin(), size.end()) - size.begin();
  std::vector<char> in(sym.n);
  for (std::uint64_t i = 0; i < sym.n; ++i)
    in[i] = comp[i] == best;
  std::fprintf(stderr, "largest weakly connected component: %llu of %llu "
                       "nodes\n",
               (unsigned long long)size[best], (unsigned long long)sym.n);
  return in;
}

static std::vector<Edge> symmetrize(std::vector<Edge> e) {
  const std::size_t m0 = e.size();
  e.reserve(2 * m0);
  for (std::size_t i = 0; i < m0; ++i)
    e.push_back({e[i].v, e[i].u, e[i].w});
  normalize(e);
  // (u,v) and (v,u) may carry different weights; give both the smaller one.
  std::vector<Edge> rev(e.size());
  for (std::size_t i = 0; i < e.size(); ++i)
    rev[i] = {e[i].v, e[i].u, e[i].w};
  std::sort(rev.begin(), rev.end(), edge_less);
  for (std::size_t i = 0; i < e.size(); ++i)
    e[i].w = std::min(e[i].w, rev[i].w); // same (u,v) order in both lists
  return e;
}

int main(int argc, char** argv) {
  std::string name, out_dir, format, weights, coords_path, coords_format,
      input;
  std::uint64_t wmin = 1, wmax = 100, seed = 1, num_sources = 10,
                num_pairs = 10;
  bool one_based = false;
  double bin_unit_m = 1.0;

  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    const auto next = [&]() -> std::string {
      if (i + 1 >= argc)
        usage(argv[0]);
      return argv[++i];
    };
    if (a == "--name")
      name = next();
    else if (a == "--out")
      out_dir = next();
    else if (a == "--format")
      format = next();
    else if (a == "--weights")
      weights = next();
    else if (a == "--wmin")
      wmin = std::stoull(next());
    else if (a == "--wmax")
      wmax = std::stoull(next());
    else if (a == "--seed")
      seed = std::stoull(next());
    else if (a == "--one-based")
      one_based = true;
    else if (a == "--coords")
      coords_path = next();
    else if (a == "--coords-format")
      coords_format = next();
    else if (a == "--bin-unit-m")
      bin_unit_m = std::stod(next());
    else if (a == "--num-sources")
      num_sources = std::stoull(next());
    else if (a == "--num-pairs")
      num_pairs = std::stoull(next());
    else if (!a.empty() && a[0] == '-')
      usage(argv[0]);
    else if (input.empty())
      input = a;
    else
      usage(argv[0]);
  }
  if (name.empty() || out_dir.empty() || input.empty() ||
      (format != "dimacs" && format != "mtx" && format != "snap"))
    usage(argv[0]);
  if (weights.empty())
    weights = format == "dimacs" ? "input" : "random";
  if (weights != "input" && weights != "random" && weights != "haversine")
    usage(argv[0]);
  if (!coords_path.empty() && coords_format != "dimacs-co" &&
      coords_format != "xyz")
    usage(argv[0]);
  if (weights == "haversine" && coords_path.empty())
    die("--weights haversine needs --coords");
  if (wmin == 0 || wmin > wmax || wmax > 0xFFFFFFFFull)
    die("need 1 <= wmin <= wmax < 2^32");
  const std::string base = out_dir + "/" + name;

  // ---------------- read edges ----------------
  std::vector<Edge> edges;
  std::uint64_t n = 0;
  bool mtx_symmetric = false;
  {
    Input in(input);
    char* line = nullptr;
    std::size_t cap = 0;
    unsigned long long lineno = 0;
    bool have_size = false;
    const bool base1 = format != "snap" || one_based;
    while (getline(&line, &cap, in.f) != -1) {
      ++lineno;
      const char c0 = line[0];
      if (format == "mtx" && lineno == 1) {
        if (std::strncmp(line, "%%MatrixMarket", 14) != 0 ||
            !std::strstr(line, "coordinate"))
          die("not a coordinate Matrix Market file", lineno);
        mtx_symmetric = std::strstr(line, "symmetric") != nullptr;
        continue;
      }
      if (c0 == '\n' || c0 == '\r' || c0 == '\0' || c0 == '%' || c0 == '#' ||
          c0 == 'c')
        continue;
      char* p = line;
      if (format == "dimacs") {
        if (c0 == 'p') {
          unsigned long long nn, mm;
          if (std::sscanf(line, "p sp %llu %llu", &nn, &mm) != 2)
            die("malformed 'p' line", lineno);
          n = nn;
          edges.reserve(mm);
          have_size = true;
          continue;
        }
        if (c0 != 'a')
          continue;
        if (!have_size)
          die("'a' line before 'p' line", lineno);
        ++p;
      }
      char* end;
      const std::uint64_t a = std::strtoull(p, &end, 10);
      if (end == p)
        die("malformed line", lineno);
      p = end;
      const std::uint64_t b = std::strtoull(p, &end, 10);
      if (end == p)
        die("malformed line", lineno);
      p = end;
      if (format == "mtx" && !have_size) {
        n = std::max(a, b);
        const std::uint64_t nnz = std::strtoull(p, &end, 10);
        edges.reserve(mtx_symmetric ? 2 * nnz : nnz);
        have_size = true;
        continue;
      }
      std::uint64_t w = 1;
      if (weights == "input") {
        const double x = std::strtod(p, &end);
        if (end == p)
          die("--weights input but the line has no weight", lineno);
        if (!(x >= 1 && x <= 4294967295.0))
          die("weight must be in [1, 2^32)", lineno);
        w = static_cast<std::uint64_t>(x);
      }
      if (base1 && (a == 0 || b == 0))
        die("id 0 in 1-based input", lineno);
      const std::uint64_t u = base1 ? a - 1 : a, v = base1 ? b - 1 : b;
      if (u >= UINT32_MAX || v >= UINT32_MAX)
        die("node id does not fit in 32 bits", lineno);
      if (format != "snap" && (u >= n || v >= n))
        die("node id out of range", lineno);
      if (format == "snap")
        n = std::max(n, std::max(u, v) + 1);
      edges.push_back({static_cast<std::uint32_t>(u),
                       static_cast<std::uint32_t>(v),
                       static_cast<std::uint32_t>(w)});
      if (mtx_symmetric)
        edges.push_back({static_cast<std::uint32_t>(v),
                         static_cast<std::uint32_t>(u),
                         static_cast<std::uint32_t>(w)});
    }
    std::free(line);
    if (format != "snap" && !have_size)
      die("no size line found");
  }
  const std::uint64_t m_input = edges.size();
  std::fprintf(stderr, "read %llu nodes, %llu edges\n", (unsigned long long)n,
               (unsigned long long)m_input);

  // ---------------- coordinates ----------------
  std::vector<double> lat, lon; // radians
  if (!coords_path.empty()) {
    lat.assign(n, NAN);
    lon.assign(n, NAN);
    Input in(coords_path);
    char* line = nullptr;
    std::size_t cap = 0;
    std::uint64_t row = 0;
    const double to_rad = kPi / 180;
    while (getline(&line, &cap, in.f) != -1) {
      double x, y;
      unsigned long long id;
      if (coords_format == "dimacs-co") {
        if (line[0] != 'v')
          continue;
        if (std::sscanf(line, "v %llu %lf %lf", &id, &x, &y) != 3 || id < 1 ||
            id > n)
          die("bad coordinate line");
        lon[id - 1] = x / 1e6 * to_rad;
        lat[id - 1] = y / 1e6 * to_rad;
      } else {
        if (std::sscanf(line, "%lf %lf", &x, &y) != 2)
          continue;
        if (row >= n)
          die("more coordinate rows than nodes");
        lon[row] = x / 1e5 * to_rad;
        lat[row] = y / 1e5 * to_rad;
        ++row;
      }
    }
    std::free(line);
    for (std::uint64_t i = 0; i < n; ++i)
      if (std::isnan(lat[i]))
        die("node " + std::to_string(i) + " has no coordinates");
  }

  // ---------------- weights ----------------
  if (weights == "random") {
    for (Edge& e : edges)
      e.w = random_weight(e.u, e.v, seed, wmin, wmax);
  } else if (weights == "haversine") {
    for (Edge& e : edges) {
      const double m =
          central_angle(lat[e.u], lon[e.u], lat[e.v], lon[e.v]) * kEarthRadiusM;
      e.w = static_cast<std::uint32_t>(std::max(1.0, std::round(m)));
    }
  }

  normalize(edges);
  const CSR g = build_csr(n, edges);
  const std::uint64_t m = g.dst.size();
  bool weights_match = false;
  const bool sym = is_symmetric(g, weights_match);
  std::fprintf(stderr, "cleaned: %llu edges (%llu self loops/duplicates "
                       "removed); %s%s\n",
               (unsigned long long)m, (unsigned long long)(m_input - m),
               sym ? "symmetric" : "directed",
               sym && !weights_match ? " (but weights differ by direction)"
                                     : "");
  constexpr std::uint64_t int32_max = std::numeric_limits<std::int32_t>::max();
  if (n > int32_max || m > int32_max)
    std::fprintf(stderr, "Warning: over 2^31-1 nodes or edges; Gunrock, "
                         "MLMQ and ece496 cannot load this graph\n");

  // ---------------- graph files ----------------
  write_gr(base + ".gr", g);
  verify_gr(base + ".gr", g);
  std::fprintf(stderr, "wrote %s.gr (verified)\n", base.c_str());

  CSR sg; // symmetric graph, only materialized when g is directed
  if (!sym || !weights_match) {
    if (!sym) {
      std::vector<Edge> t(edges.size());
      for (std::size_t i = 0; i < edges.size(); ++i)
        t[i] = {edges[i].v, edges[i].u, edges[i].w};
      std::sort(t.begin(), t.end(), edge_less);
      const CSR tg = build_csr(n, t);
      write_gr(base + ".tgr", tg);
      verify_gr(base + ".tgr", tg);
      std::fprintf(stderr, "wrote %s.tgr (verified)\n", base.c_str());
    }
    sg = build_csr(n, symmetrize(edges));
    write_gr(base + ".sgr", sg);
    verify_gr(base + ".sgr", sg);
    std::fprintf(stderr, "wrote %s.sgr (verified, %llu edges)\n", base.c_str(),
                 (unsigned long long)sg.dst.size());
  }
  std::vector<Edge>().swap(edges);
  const CSR& symg = (sym && weights_match) ? g : sg;

  if (sym && weights_match) {
    write_mtx(base + ".mtx", g, true);
  } else {
    write_mtx(base + ".mtx", g, false);
    write_mtx(base + ".sym.mtx", sg, true);
  }
  std::fprintf(stderr, "wrote Matrix Market file(s)\n");
  write_ag(base + ".sym.ag", symg);
  std::fprintf(stderr, "wrote %s.sym.ag\n", base.c_str());

  std::uint64_t clamped = 0;
  if (!lat.empty()) {
    clamped = write_bin(base + ".bin", g, lat, lon, bin_unit_m);
    std::fprintf(stderr, "wrote %s.bin (%llu of %llu edges raised to the "
                         "straight-line distance)\n",
                 base.c_str(), (unsigned long long)clamped,
                 (unsigned long long)m);
  }

  // ---------------- sources, pairs, references ----------------
  const std::vector<char> in_wcc = largest_wcc(symg);
  std::uint32_t main_src = 0;
  while (main_src < n && !(in_wcc[main_src] && g.off[main_src + 1] >
                                                   g.off[main_src]))
    ++main_src;
  if (main_src == n)
    die("no usable source vertex");

  std::mt19937_64 rng(seed);
  const std::vector<std::uint32_t> main_level = bfs(g, main_src);
  std::vector<std::uint32_t> pool; // reached from main_src, has out-edges
  for (std::uint64_t i = 0; i < n; ++i)
    if (main_level[i] != UINT32_MAX && g.off[i + 1] > g.off[i] &&
        i != main_src)
      pool.push_back(i);
  std::vector<std::uint32_t> sources{main_src};
  num_sources = std::min<std::uint64_t>(num_sources, pool.size());
  while (sources.size() < num_sources + 1) {
    const std::uint32_t s = pool[rng() % pool.size()];
    if (std::find(sources.begin(), sources.end(), s) == sources.end())
      sources.push_back(s);
  }

  {
    Writer f(out_dir + "/sources.txt");
    for (std::uint32_t s : sources) {
      f.num(s);
      f.ch('\n');
    }
  }

  struct Pair {
    std::uint32_t s, t;
    std::uint64_t dist;
    std::uint32_t hops;
  };
  std::vector<Pair> random_pairs, far_pairs;
  Writer ref(out_dir + "/reference.txt");
  ref.str(("# Reference results on " + name +
           ".gr (0-based ids, integer weights)\n"
           "# source bfs_reached bfs_max_level bfs_sum_levels sssp_reached "
           "sssp_max_dist sssp_sum_dist\n")
              .c_str());
  for (std::size_t k = 0; k < sources.size(); ++k) {
    const std::uint32_t s = sources[k];
    const std::vector<std::uint32_t> level = k == 0 ? main_level : bfs(g, s);
    const std::vector<std::uint64_t> dist = dijkstra(g, s);
    const Summary b = summarize(level), d = summarize(dist);
    for (std::uint64_t x : {std::uint64_t{s}, b.reached, b.max, b.sum,
                            d.reached, d.max, d.sum}) {
      ref.num(x);
      ref.ch(' ');
    }
    ref.ch('\n');
    std::fprintf(stderr, "source %u: reaches %llu nodes, max level %llu, max "
                         "dist %llu\n",
                 s, (unsigned long long)b.reached, (unsigned long long)b.max,
                 (unsigned long long)d.max);
    if (k == 0 || random_pairs.size() >= num_pairs)
      continue;
    // One random reachable target and the farthest target from this source.
    std::vector<std::uint32_t> reach;
    std::uint32_t far = s;
    for (std::uint64_t i = 0; i < n; ++i)
      if (dist[i] != UINT64_MAX && i != s) {
        reach.push_back(i);
        if (dist[i] > dist[far])
          far = i;
      }
    if (reach.empty())
      continue;
    const std::uint32_t t = reach[rng() % reach.size()];
    random_pairs.push_back({s, t, dist[t], level[t]});
    far_pairs.push_back({s, far, dist[far], level[far]});
  }

  ref.str("# pairs (same order as pairs.txt): source target sssp_dist "
          "bfs_hops\n");
  {
    Writer pf(out_dir + "/pairs.txt");
    for (const auto* list : {&random_pairs, &far_pairs})
      for (const Pair& p : *list) {
        pf.num(p.s);
        pf.ch(' ');
        pf.num(p.t);
        pf.ch('\n');
        for (std::uint64_t x : {std::uint64_t{p.s}, std::uint64_t{p.t},
                                p.dist, std::uint64_t{p.hops}}) {
          ref.num(x);
          ref.ch(' ');
        }
        ref.ch('\n');
      }
  }

  // ---------------- info ----------------
  Writer info(out_dir + "/info.txt");
  const auto kv = [&](const char* k, const std::string& v) {
    info.str(k);
    info.str(": ");
    info.str(v.c_str());
    info.ch('\n');
  };
  std::uint64_t wlo = UINT64_MAX, whi = 0;
  for (std::uint32_t x : g.w) {
    wlo = std::min<std::uint64_t>(wlo, x);
    whi = std::max<std::uint64_t>(whi, x);
  }
  kv("name", name);
  kv("input", input);
  kv("input_format", format);
  kv("coords", coords_path.empty() ? "none" : coords_path + " (" +
                                                  coords_format + ")");
  kv("nodes", std::to_string(n));
  kv("edges", std::to_string(m));
  kv("input_edges", std::to_string(m_input));
  kv("directed", sym ? "no (every edge stored in both directions)" : "yes");
  if (!sym || !weights_match)
    kv("symmetric_edges", std::to_string(sg.dst.size()) +
                              " (in .sgr / .sym.mtx / .sym.ag; weight = min "
                              "of both directions)");
  kv("weights", weights == "random"
                    ? "random [" + std::to_string(wmin) + ", " +
                          std::to_string(wmax) + "], seed " +
                          std::to_string(seed)
                    : weights == "haversine"
                          ? "great-circle metres from coordinates"
                          : "from input");
  kv("weight_range", std::to_string(wlo) + " " + std::to_string(whi));
  if (!lat.empty())
    kv("bin_edge_length", "max(weight * " + std::to_string(bin_unit_m) +
                              " m, straight line + 1 cm); " +
                              std::to_string(clamped) + " edges raised");
  kv("main_source", std::to_string(main_src));
  kv("seed", std::to_string(seed));
  return 0;
}
