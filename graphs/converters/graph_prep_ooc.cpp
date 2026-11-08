// graph_prep_ooc.cpp
// Out-of-core version of graph_prep for edge lists too large to sort in RAM
// (Friendster, Web Data Commons hyperlink graphs). Input is a plain edge list
// ("u v" per line, '#'/'%' comments, 0-based unless --one-based), plain or
// compressed. Weights are always the random hash shared with graph_prep, so
// a graph converted by either tool is byte-identical.
//
// Outputs in --out (see graph_prep.cpp for the formats):
//   G.gr                     always (symmetrized with --symmetrize)
//   G.tgr, G.sgr             only for directed graphs
//   G.sym.ag                 always (PBBS text of the symmetric graph)
//   G.mtx                    only with --mtx ("general" storage)
//   orig_ids.bin             only with --compact-ids: u32 original id per node
//   sources.txt, pairs.txt, reference.txt, info.txt
//
// How it works:
//   1. One pass over the input writes the edges as raw u32 pairs to
//      <work>/raw.bin and counts in/out degrees.
//   2. For each output graph, edges are bucketed by source node range into
//      partition files sized to --budget-edges, then each partition is sorted
//      and deduplicated in RAM and appended to the output.
//   3. References: the final .gr is loaded (offsets + destinations, weights
//      recomputed from the hash), BFS runs from every source and Dijkstra from
//      the main source.
//
// Peak RAM is about 8 bytes x budget-edges for sorting, then about
// 4 bytes x edges + 20 bytes x nodes for the references (Friendster: ~16 GB).
// Temporary disk use is about 3x the final .gr size.
//
// Build: g++ -O3 -std=c++17 -o graph_prep_ooc graph_prep_ooc.cpp
// Usage: ./graph_prep_ooc --name G --out DIR --work DIR [options] <input>

#include <algorithm>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <limits>
#include <queue>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "graph_prep_common.h"

static void usage(const char* prog) {
  std::fprintf(
      stderr,
      "Usage: %s --name G --out DIR --work DIR [options] <input>\n"
      "  --symmetrize            input lists each undirected edge once; store "
      "both directions\n"
      "  --compact-ids           renumber nodes that have edges to 0..n-1 "
      "(keeps order)\n"
      "  --mtx                   also write G.mtx (Gunrock)\n"
      "  --one-based             input ids start at 1\n"
      "  --wmin N --wmax N       random weight range (default 1 100)\n"
      "  --seed N                seed for weights and sources (default 1)\n"
      "  --budget-edges N        edges sorted in RAM at once (default 1e9)\n"
      "  --num-sources N         random sources besides the main one "
      "(default 10)\n"
      "  --num-pairs N           random + N farthest pairs (default 10)\n",
      prog);
  std::exit(1);
}

static void note(const char* fmt, ...) __attribute__((format(printf, 1, 2)));
static void note(const char* fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  std::vfprintf(stderr, fmt, ap);
  va_end(ap);
  std::fputc('\n', stderr);
  std::fflush(stderr);
}

static std::FILE* open_or_die(const std::string& path, const char* mode) {
  std::FILE* f = std::fopen(path.c_str(), mode);
  if (!f)
    die("cannot open " + path);
  return f;
}

static void read_or_die(std::FILE* f, void* p, std::size_t size,
                        std::size_t count, const char* what) {
  if (count && std::fread(p, size, count, f) != count)
    die(std::string("short read: ") + what);
}

// ---------------------------------------------------------------------------
// Pass 1: parse the edge list into raw u32 pairs and count degrees.

struct Degrees {
  std::vector<std::uint32_t> out, in; // indexed by original id
  void touch(std::uint64_t id) {
    if (id >= out.size()) {
      const std::size_t n = std::max<std::size_t>(id + 1, out.size() * 3 / 2);
      out.resize(n, 0);
      in.resize(n, 0);
    }
  }
};

static std::uint64_t read_edges(const std::string& input, bool one_based,
                                const std::string& raw_path, Degrees& deg,
                                std::uint64_t& max_id,
                                std::uint64_t& self_loops) {
  Input in(input);
  Writer raw(raw_path, 1 << 24);
  std::vector<char> buf(1 << 24);
  std::uint64_t lines = 0, edges = 0;
  max_id = 0;
  self_loops = 0;

  // Line state machine that survives chunk boundaries.
  bool at_line_start = true, skipping = false, in_num = false;
  int field = 0; // numbers seen on this line
  std::uint64_t cur = 0, a = 0;
  const auto end_line = [&]() {
    at_line_start = true;
    skipping = false;
    in_num = false;
    field = 0;
    ++lines;
    if (lines % 200000000 == 0)
      note("  read %llu lines", (unsigned long long)lines);
  };
  const auto finish_num = [&]() {
    in_num = false;
    if (field == 0) {
      a = cur;
      field = 1;
    } else if (field == 1) {
      std::uint64_t u = a, v = cur;
      field = 2;
      skipping = true; // ignore anything else on the line
      if (one_based) {
        if (u == 0 || v == 0)
          die("id 0 in 1-based input", lines + 1);
        --u;
        --v;
      }
      if (u >= UINT32_MAX || v >= UINT32_MAX)
        die("node id does not fit in 32 bits", lines + 1);
      if (u == v) {
        ++self_loops;
        return;
      }
      deg.touch(std::max(u, v));
      ++deg.out[u];
      ++deg.in[v];
      max_id = std::max(max_id, std::max(u, v));
      raw.bin(static_cast<std::uint32_t>(u));
      raw.bin(static_cast<std::uint32_t>(v));
      ++edges;
    }
  };

  std::size_t got;
  while ((got = std::fread(buf.data(), 1, buf.size(), in.f)) > 0) {
    for (std::size_t i = 0; i < got; ++i) {
      const char c = buf[i];
      if (c == '\n') {
        if (in_num)
          finish_num();
        if (!at_line_start && field == 1)
          die("line with a single number", lines + 1);
        end_line();
        continue;
      }
      if (skipping)
        continue;
      if (at_line_start) {
        at_line_start = false;
        if (c == '#' || c == '%') {
          skipping = true;
          continue;
        }
      }
      if (c >= '0' && c <= '9') {
        if (!in_num) {
          in_num = true;
          cur = 0;
        }
        cur = cur * 10 + static_cast<std::uint64_t>(c - '0');
      } else if (c == ' ' || c == '\t' || c == '\r' || c == ',') {
        if (in_num)
          finish_num();
      } else {
        die(std::string("unexpected character '") + c + "'", lines + 1);
      }
    }
  }
  if (in_num)
    finish_num();
  if (!at_line_start && field == 1)
    die("line with a single number", lines + 1);
  return edges;
}

// ---------------------------------------------------------------------------
// Building one .gr out of core.

enum class Mode { Out, In, Both }; // (u,v) | (v,u) | both

struct Ctx {
  std::string work;
  std::uint64_t n = 0;
  std::uint64_t budget = 1000000000;
  std::uint64_t seed = 1, wmin = 1, wmax = 100;
  std::vector<std::uint32_t> new_id; // original id -> node id (empty = same)
  std::vector<std::uint32_t> deg_out, deg_in; // by node id
  std::uint32_t map(std::uint32_t x) const {
    return new_id.empty() ? x : new_id[x];
  }
};

// Streams raw.bin in chunks, calling f(u, v) with node ids.
template <typename F>
static void for_each_raw(const Ctx& c, F&& f) {
  std::FILE* in = open_or_die(c.work + "/raw.bin", "rb");
  std::vector<std::uint32_t> buf(1 << 23);
  std::size_t got;
  while ((got = std::fread(buf.data(), 4, buf.size(), in)) > 0) {
    if (got % 2)
      die("raw.bin is corrupt");
    for (std::size_t i = 0; i < got; i += 2)
      f(c.map(buf[i]), c.map(buf[i + 1]));
  }
  std::fclose(in);
}

// Writes a .gr for the chosen orientation and returns its edge count.
static std::uint64_t build_gr(const Ctx& c, Mode mode,
                              const std::string& path) {
  // Upper bound on each node's out-degree in this orientation.
  const auto key_count = [&](std::uint64_t x) -> std::uint64_t {
    switch (mode) {
    case Mode::Out:
      return c.deg_out[x];
    case Mode::In:
      return c.deg_in[x];
    default:
      return std::uint64_t{c.deg_out[x]} + c.deg_in[x];
    }
  };

  // Contiguous node ranges, each with at most `budget` edges.
  std::vector<std::uint64_t> bound{0}; // partition p = [bound[p], bound[p+1])
  std::uint64_t acc = 0;
  for (std::uint64_t x = 0; x < c.n; ++x) {
    const std::uint64_t k = key_count(x);
    if (k > c.budget)
      die("a single node has more edges than --budget-edges");
    if (acc + k > c.budget) {
      bound.push_back(x);
      acc = 0;
    }
    acc += k;
  }
  bound.push_back(c.n);
  const std::size_t parts = bound.size() - 1;
  note("  %s: %zu partition(s)", path.c_str(), parts);

  const auto part_of = [&](std::uint32_t x) {
    return static_cast<std::size_t>(
               std::upper_bound(bound.begin(), bound.end(), x) -
               bound.begin()) -
           1;
  };

  {
    std::vector<Writer*> pw;
    for (std::size_t p = 0; p < parts; ++p)
      pw.push_back(
          new Writer(c.work + "/part" + std::to_string(p) + ".bin", 1 << 23));
    const auto emit = [&](std::uint32_t key, std::uint32_t val) {
      pw[parts == 1 ? 0 : part_of(key)]->bin(
          (std::uint64_t{key} << 32) | val);
    };
    for_each_raw(c, [&](std::uint32_t u, std::uint32_t v) {
      if (mode != Mode::In)
        emit(u, v);
      if (mode != Mode::Out)
        emit(v, u);
    });
    for (Writer* w : pw)
      delete w;
  }

  // Sort and dedup each partition; offsets stay in RAM, destinations go to
  // dst.tmp.
  std::vector<std::uint64_t> end_off(c.n, 0);
  std::uint64_t m = 0;
  {
    Writer dst(c.work + "/dst.tmp", 1 << 24);
    std::vector<std::uint64_t> e;
    for (std::size_t p = 0; p < parts; ++p) {
      const std::string pp = c.work + "/part" + std::to_string(p) + ".bin";
      std::FILE* f = open_or_die(pp, "rb");
      std::fseek(f, 0, SEEK_END);
      const std::size_t cnt = std::ftell(f) / 8;
      std::fseek(f, 0, SEEK_SET);
      e.resize(cnt);
      read_or_die(f, e.data(), 8, cnt, "partition");
      std::fclose(f);
      std::remove(pp.c_str());
      std::sort(e.begin(), e.end());
      e.erase(std::unique(e.begin(), e.end()), e.end());
      std::size_t i = 0;
      for (std::uint64_t x = bound[p]; x < bound[p + 1]; ++x) {
        while (i < e.size() && (e[i] >> 32) == x) {
          dst.bin(static_cast<std::uint32_t>(e[i]));
          ++i;
          ++m;
        }
        end_off[x] = m;
      }
      if (i != e.size())
        die("partition bookkeeping error");
    }
    std::vector<std::uint64_t>().swap(e);
  }

  // Assemble: header, offsets, destinations, pad, weights.
  {
    Writer out(path, 1 << 24);
    const std::uint64_t header[4] = {1, 4, c.n, m};
    out.raw(header, sizeof(header));
    out.raw(end_off.data(), 8 * c.n);
    std::vector<std::uint32_t> buf(1 << 23);
    for (int pass = 0; pass < 2; ++pass) {
      std::FILE* d = open_or_die(c.work + "/dst.tmp", "rb");
      std::uint64_t idx = 0, node = 0;
      std::size_t got;
      while ((got = std::fread(buf.data(), 4, buf.size(), d)) > 0) {
        if (pass == 0) {
          out.raw(buf.data(), 4 * got);
        } else {
          for (std::size_t i = 0; i < got; ++i, ++idx) {
            while (end_off[node] <= idx)
              ++node;
            buf[i] = random_weight(static_cast<std::uint32_t>(node), buf[i],
                                   c.seed, c.wmin, c.wmax);
          }
          out.raw(buf.data(), 4 * got);
        }
      }
      std::fclose(d);
      if (pass == 0 && m % 2)
        out.bin(std::uint32_t{0});
    }
  }
  std::remove((c.work + "/dst.tmp").c_str());

  // Size check.
  std::FILE* f = open_or_die(path, "rb");
  std::fseek(f, 0, SEEK_END);
  const std::uint64_t size = std::ftell(f);
  std::fclose(f);
  if (size != 32 + 8 * c.n + 4 * (m + (m & 1)) + 4 * m)
    die("size check failed for " + path);
  note("  wrote %s (%llu edges)", path.c_str(), (unsigned long long)m);
  return m;
}

// ---------------------------------------------------------------------------
// Streaming conversions from a finished .gr.

struct GrReader {
  std::uint64_t n = 0, m = 0;
  std::vector<std::uint64_t> end_off;
  std::FILE* dst = nullptr;
  std::FILE* wt = nullptr;
  explicit GrReader(const std::string& path, bool weights) {
    dst = open_or_die(path, "rb");
    std::uint64_t h[4];
    read_or_die(dst, h, 8, 4, "gr header");
    if (h[0] != 1 || h[1] != 4)
      die("unexpected .gr header in " + path);
    n = h[2];
    m = h[3];
    end_off.resize(n);
    read_or_die(dst, end_off.data(), 8, n, "gr offsets");
    if (weights) {
      wt = open_or_die(path, "rb");
      std::fseek(wt, 32 + 8 * n + 4 * (m + (m & 1)), SEEK_SET);
    }
  }
  ~GrReader() {
    std::fclose(dst);
    if (wt)
      std::fclose(wt);
  }
};

static void write_ag(const std::string& gr, const std::string& path) {
  GrReader g(gr, false);
  Writer out(path, 1 << 24);
  out.str("AdjacencyGraph\n");
  out.num(g.n);
  out.ch('\n');
  out.num(g.m);
  out.ch('\n');
  for (std::uint64_t u = 0; u < g.n; ++u) {
    out.num(u ? g.end_off[u - 1] : 0);
    out.ch('\n');
  }
  std::vector<std::uint32_t> buf(1 << 23);
  for (std::uint64_t left = g.m; left;) {
    const std::size_t k = std::min<std::uint64_t>(left, buf.size());
    read_or_die(g.dst, buf.data(), 4, k, "gr destinations");
    for (std::size_t i = 0; i < k; ++i) {
      out.num(buf[i]);
      out.ch('\n');
    }
    left -= k;
  }
  note("  wrote %s", path.c_str());
}

static void write_mtx(const std::string& gr, const std::string& path) {
  GrReader g(gr, true);
  Writer out(path, 1 << 24);
  out.str("%%MatrixMarket matrix coordinate integer general\n");
  out.num(g.n);
  out.ch(' ');
  out.num(g.n);
  out.ch(' ');
  out.num(g.m);
  out.ch('\n');
  std::vector<std::uint32_t> d(1 << 22), w(1 << 22);
  std::uint64_t idx = 0, node = 0;
  for (std::uint64_t left = g.m; left;) {
    const std::size_t k = std::min<std::uint64_t>(left, d.size());
    read_or_die(g.dst, d.data(), 4, k, "gr destinations");
    read_or_die(g.wt, w.data(), 4, k, "gr weights");
    for (std::size_t i = 0; i < k; ++i, ++idx) {
      while (g.end_off[node] <= idx)
        ++node;
      out.num(node + 1);
      out.ch(' ');
      out.num(std::uint64_t{d[i]} + 1);
      out.ch(' ');
      out.num(w[i]);
      out.ch('\n');
    }
    left -= k;
  }
  note("  wrote %s", path.c_str());
}

// ---------------------------------------------------------------------------
// References on the main graph, held in RAM without its weights.

struct Graph {
  std::uint64_t n = 0;
  std::vector<std::uint64_t> off; // n + 1
  std::vector<std::uint32_t> dst;
};

static Graph load_structure(const std::string& gr) {
  GrReader r(gr, false);
  Graph g;
  g.n = r.n;
  g.off.resize(r.n + 1);
  g.off[0] = 0;
  std::copy(r.end_off.begin(), r.end_off.end(), g.off.begin() + 1);
  g.dst.resize(r.m);
  read_or_die(r.dst, g.dst.data(), 4, r.m, "gr destinations");
  return g;
}

struct Summary {
  std::uint64_t reached = 0, max = 0, sum = 0;
};

static std::vector<std::uint32_t> bfs(const Graph& g, std::uint32_t s,
                                      Summary& sum) {
  std::vector<std::uint32_t> level(g.n, UINT32_MAX), q;
  q.reserve(1 << 24);
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
  sum = {};
  for (std::uint32_t x : level)
    if (x != UINT32_MAX) {
      ++sum.reached;
      sum.max = std::max<std::uint64_t>(sum.max, x);
      sum.sum += x;
    }
  return level;
}

static std::vector<std::uint64_t> dijkstra(const Graph& g, const Ctx& c,
                                           std::uint32_t s, Summary& sum) {
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
      const std::uint32_t v = g.dst[i];
      const std::uint64_t nd = d + random_weight(u, v, c.seed, c.wmin, c.wmax);
      if (nd < dist[v]) {
        dist[v] = nd;
        pq.push({nd, v});
      }
    }
  }
  sum = {};
  for (std::uint64_t x : dist)
    if (x != UINT64_MAX) {
      ++sum.reached;
      sum.max = std::max(sum.max, x);
      sum.sum += x;
    }
  return dist;
}

// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
  std::string name, out_dir, work, input;
  bool symmetrize = false, compact = false, mtx = false, one_based = false;
  std::uint64_t num_sources = 10, num_pairs = 10;
  Ctx c;

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
    else if (a == "--work")
      work = next();
    else if (a == "--symmetrize")
      symmetrize = true;
    else if (a == "--compact-ids")
      compact = true;
    else if (a == "--mtx")
      mtx = true;
    else if (a == "--one-based")
      one_based = true;
    else if (a == "--wmin")
      c.wmin = std::stoull(next());
    else if (a == "--wmax")
      c.wmax = std::stoull(next());
    else if (a == "--seed")
      c.seed = std::stoull(next());
    else if (a == "--budget-edges")
      c.budget = std::stoull(next());
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
  if (name.empty() || out_dir.empty() || work.empty() || input.empty() ||
      c.budget == 0)
    usage(argv[0]);
  if (c.wmin == 0 || c.wmin > c.wmax || c.wmax > 0xFFFFFFFFull)
    die("need 1 <= wmin <= wmax < 2^32");
  c.work = work;
  const std::string base = out_dir + "/" + name;

  // ---- pass 1 ----
  note("reading %s", input.c_str());
  Degrees deg;
  std::uint64_t max_id = 0, self_loops = 0;
  const std::uint64_t raw_edges =
      read_edges(input, one_based, work + "/raw.bin", deg, max_id, self_loops);
  const std::uint64_t id_space = raw_edges ? max_id + 1 : 0;
  deg.out.resize(id_space);
  deg.in.resize(id_space);
  note("read %llu edges (%llu self loops dropped), ids 0..%llu",
      (unsigned long long)raw_edges, (unsigned long long)self_loops,
      (unsigned long long)max_id);

  // ---- node ids ----
  if (compact) {
    c.new_id.assign(id_space, UINT32_MAX);
    Writer orig(out_dir + "/orig_ids.bin");
    std::uint64_t next_id = 0;
    for (std::uint64_t x = 0; x < id_space; ++x)
      if (deg.out[x] || deg.in[x]) {
        c.new_id[x] = static_cast<std::uint32_t>(next_id++);
        orig.bin(static_cast<std::uint32_t>(x));
        c.deg_out.push_back(deg.out[x]);
        c.deg_in.push_back(deg.in[x]);
      }
    c.n = next_id;
    note("compacted ids: %llu nodes with edges out of %llu ids",
        (unsigned long long)c.n, (unsigned long long)id_space);
  } else {
    c.n = id_space;
    c.deg_out = std::move(deg.out);
    c.deg_in = std::move(deg.in);
  }
  std::vector<std::uint32_t>().swap(deg.out);
  std::vector<std::uint32_t>().swap(deg.in);

  // ---- graphs ----
  const std::uint64_t m =
      build_gr(c, symmetrize ? Mode::Both : Mode::Out, base + ".gr");
  std::uint64_t m_sym = m;
  if (!symmetrize) {
    build_gr(c, Mode::In, base + ".tgr");
    m_sym = build_gr(c, Mode::Both, base + ".sgr");
  }
  std::remove((work + "/raw.bin").c_str());
  constexpr std::uint64_t int32_max = std::numeric_limits<std::int32_t>::max();
  if (c.n > int32_max || m > int32_max)
    note("Warning: over 2^31-1 nodes or edges; Gunrock, MLMQ and ece496 "
        "cannot load %s.gr",
        name.c_str());

  write_ag(symmetrize ? base + ".gr" : base + ".sgr", base + ".sym.ag");
  if (mtx)
    write_mtx(base + ".gr", base + ".mtx");

  // ---- sources, pairs, references ----
  note("loading %s.gr for references", base.c_str());
  std::vector<std::uint32_t>().swap(c.deg_out);
  std::vector<std::uint32_t>().swap(c.deg_in);
  std::vector<std::uint32_t>().swap(c.new_id);
  const Graph g = load_structure(base + ".gr");

  // Main source: the node with the most out-edges (smallest id on ties).
  std::uint32_t main_src = 0;
  for (std::uint64_t x = 1; x < g.n; ++x)
    if (g.off[x + 1] - g.off[x] > g.off[main_src + 1] - g.off[main_src])
      main_src = x;

  std::mt19937_64 rng(c.seed);
  Summary main_bfs;
  const std::vector<std::uint32_t> main_level = bfs(g, main_src, main_bfs);
  std::vector<std::uint32_t> pool;
  for (std::uint64_t x = 0; x < g.n; ++x)
    if (main_level[x] != UINT32_MAX && g.off[x + 1] > g.off[x] &&
        x != main_src)
      pool.push_back(x);
  std::vector<std::uint32_t> sources{main_src};
  num_sources = std::min<std::uint64_t>(num_sources, pool.size());
  while (sources.size() < num_sources + 1) {
    const std::uint32_t s = pool[rng() % pool.size()];
    if (std::find(sources.begin(), sources.end(), s) == sources.end())
      sources.push_back(s);
  }
  std::vector<std::uint32_t>().swap(pool);
  {
    Writer f(out_dir + "/sources.txt");
    for (std::uint32_t s : sources) {
      f.num(s);
      f.ch('\n');
    }
  }

  Writer ref(out_dir + "/reference.txt");
  ref.str(("# Reference results on " + name +
           ".gr (0-based ids, integer weights)\n"
           "# source bfs_reached bfs_max_level bfs_sum_levels sssp_reached "
           "sssp_max_dist sssp_sum_dist\n"
           "# (SSSP is computed for the main source only; NA elsewhere)\n")
              .c_str());
  const auto put = [&](std::uint64_t x) {
    ref.num(x);
    ref.ch(' ');
  };
  Summary main_sssp;
  note("Dijkstra from main source %u", main_src);
  const std::vector<std::uint64_t> dist = dijkstra(g, c, main_src, main_sssp);
  for (std::size_t k = 0; k < sources.size(); ++k) {
    Summary b = main_bfs;
    if (k)
      bfs(g, sources[k], b);
    put(sources[k]);
    put(b.reached);
    put(b.max);
    put(b.sum);
    if (k == 0) {
      put(main_sssp.reached);
      put(main_sssp.max);
      put(main_sssp.sum);
    } else {
      ref.str("NA NA NA ");
    }
    ref.ch('\n');
    note("source %u: reaches %llu nodes, max level %llu", sources[k],
        (unsigned long long)b.reached, (unsigned long long)b.max);
  }

  // Pairs from the main source: random reachable targets, then the
  // farthest nodes.
  std::vector<std::uint32_t> reach;
  for (std::uint64_t x = 0; x < g.n; ++x)
    if (dist[x] != UINT64_MAX && x != main_src)
      reach.push_back(x);
  std::vector<std::uint32_t> targets;
  const std::uint64_t want = std::min<std::uint64_t>(num_pairs, reach.size());
  while (targets.size() < want) {
    const std::uint32_t t = reach[rng() % reach.size()];
    if (std::find(targets.begin(), targets.end(), t) == targets.end())
      targets.push_back(t);
  }
  {
    std::vector<std::uint32_t> far = reach;
    const std::size_t k = std::min<std::size_t>(want, far.size());
    std::partial_sort(far.begin(), far.begin() + k, far.end(),
                      [&](std::uint32_t a, std::uint32_t b) {
                        return dist[a] != dist[b] ? dist[a] > dist[b] : a < b;
                      });
    targets.insert(targets.end(), far.begin(), far.begin() + k);
  }
  ref.str("# pairs (same order as pairs.txt): source target sssp_dist "
          "bfs_hops\n");
  {
    Writer pf(out_dir + "/pairs.txt");
    for (std::uint32_t t : targets) {
      pf.num(main_src);
      pf.ch(' ');
      pf.num(t);
      pf.ch('\n');
      put(main_src);
      put(t);
      put(dist[t]);
      put(main_level[t]);
      ref.ch('\n');
    }
  }

  // ---- info ----
  Writer info(out_dir + "/info.txt");
  const auto kv = [&](const char* k, const std::string& v) {
    info.str(k);
    info.str(": ");
    info.str(v.c_str());
    info.ch('\n');
  };
  kv("name", name);
  kv("input", input);
  kv("input_format", "edgelist (out of core)");
  kv("nodes", std::to_string(c.n));
  kv("edges", std::to_string(m));
  kv("input_edges", std::to_string(raw_edges + self_loops));
  kv("directed", symmetrize ? "no (every edge stored in both directions)"
                            : "yes");
  if (!symmetrize)
    kv("symmetric_edges",
       std::to_string(m_sym) + " (in .sgr / .sym.ag; weights are symmetric)");
  if (compact)
    kv("ids", "compacted: node i is original id orig_ids.bin[i] (u32); " +
                  std::to_string(id_space - c.n) + " ids without edges dropped");
  kv("weights", "random [" + std::to_string(c.wmin) + ", " +
                    std::to_string(c.wmax) + "], seed " +
                    std::to_string(c.seed));
  kv("main_source", std::to_string(main_src) + " (highest out-degree)");
  kv("pairs", "all from the main source: random targets, then farthest");
  kv("seed", std::to_string(c.seed));
  return 0;
}
