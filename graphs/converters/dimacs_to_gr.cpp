// dimacs2gr.cpp
// Convert a DIMACS text graph ("p sp N M" / "a u v [w]", 1-based) into the
// Galois binary .gr format read by CSRGraph::readFromGR:
//   u64 version(=1), u64 sizeof(edge data)(=4), u64 numNodes, u64 numEdges
//   u64 outIdx[numNodes]     (cumulative end offset of each node's edges)
//   u32 dst[numEdges]        (0-based)
//   u32 pad (only if numEdges is odd)
//   u32 weight[numEdges]
//
// Optional node-data trailer (only written when node data is present):
//   u64 kind                 (1 = single u32 value, 2 = (i32 x, i32 y) pair)
//   kind 1: u32 value[numNodes]
//   kind 2: i32 xy[2*numNodes]   (x0,y0,x1,y1,...)
//
// Node data lines (in the graph file, or in an optional separate file such
// as a DIMACS .co file); ids are 1-based, unlisted nodes default to 0:
//   n <id> <value>           single value  (unsigned 32-bit)
//   v <id> <x> <y>           coordinate pair (signed 32-bit)
// A file may use only one of the two kinds.
//
// Two passes over the text file keep memory low:
//   pass 1 counts out-degrees, pass 2 places each edge directly in CSR order.
// Edges that have no weight field get weight 1.
//
// Build: g++ -O2 -std=c++17 -o dimacs2gr dimacs2gr.cpp
// Usage: ./dimacs2gr input.gr output.bin.gr [input.co]

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static bool is_little_endian() {
  const std::uint16_t x = 1;
  return *reinterpret_cast<const unsigned char*>(&x) == 1;
}

[[noreturn]] static void die(const char* msg, unsigned long long line = 0) {
  if (line)
    std::fprintf(stderr, "Error (line %llu): %s\n", line, msg);
  else
    std::fprintf(stderr, "Error: %s\n", msg);
  std::exit(1);
}

struct Arc {
  std::uint64_t u, v, w;
};

// Parses an 'a' line (the text after the leading 'a'). Weight defaults to 1.
static bool parse_arc(const char* p, Arc& arc) {
  char* end;
  arc.u = std::strtoull(p, &end, 10);
  if (end == p)
    return false;
  p = end;
  arc.v = std::strtoull(p, &end, 10);
  if (end == p)
    return false;
  p = end;
  arc.w = std::strtoull(p, &end, 10);
  if (end == p)
    arc.w = 1; // no weight field
  return true;
}

// ---------------------------- node data ----------------------------
enum NodeKind : std::uint64_t { NONE = 0, SINGLE = 1, PAIR = 2 };

struct NodeData {
  NodeKind kind = NONE;
  std::vector<std::uint32_t> data; // SINGLE: n values; PAIR: 2n (x,y) values
};

static void set_kind(NodeData& nd, NodeKind k, std::uint64_t n,
                     unsigned long long lineno) {
  if (nd.kind == NONE) {
    nd.kind = k;
    nd.data.assign(k == PAIR ? 2 * n : n, 0);
  } else if (nd.kind != k) {
    die("cannot mix 'n' (single value) and 'v' (coordinate) node lines",
        lineno);
  }
}

// Handles an 'n' or 'v' line (line[0] is the tag). Requires n to be known.
static void parse_node_line(const char* line, std::uint64_t n, NodeData& nd,
                            unsigned long long lineno) {
  const char* p = line + 1;
  char* end;

  const std::uint64_t id = std::strtoull(p, &end, 10);
  if (end == p)
    die("malformed node-data line (missing node id)", lineno);
  p = end;
  if (id < 1 || id > n)
    die("node id out of range 1..N in node-data line", lineno);

  if (line[0] == 'n') {
    const unsigned long long val = std::strtoull(p, &end, 10);
    if (end == p)
      die("malformed 'n' line (missing value)", lineno);
    if (val > 0xFFFFFFFFull)
      die("node value does not fit in 32 bits", lineno);
    set_kind(nd, SINGLE, n, lineno);
    nd.data[id - 1] = static_cast<std::uint32_t>(val);
  } else { // 'v'
    const long long x = std::strtoll(p, &end, 10);
    if (end == p)
      die("malformed 'v' line (missing x)", lineno);
    p = end;
    const long long y = std::strtoll(p, &end, 10);
    if (end == p)
      die("malformed 'v' line (missing y)", lineno);
    if (x < INT32_MIN || x > INT32_MAX || y < INT32_MIN || y > INT32_MAX)
      die("coordinate does not fit in signed 32 bits", lineno);
    set_kind(nd, PAIR, n, lineno);
    nd.data[2 * (id - 1)] = static_cast<std::uint32_t>(static_cast<std::int32_t>(x));
    nd.data[2 * (id - 1) + 1] = static_cast<std::uint32_t>(static_cast<std::int32_t>(y));
  }
}

int main(int argc, char** argv) {
  if (argc != 3 && argc != 4) {
    std::fprintf(stderr, "Usage: %s <input_dimacs.gr> <output_binary.gr> [node_data.txt]\n",
                 argv[0]);
    return 1;
  }
  if (!is_little_endian())
    die("big-endian hosts are not supported");

  const char* in_path = argv[1];
  const char* out_path = argv[2];
  const char* node_path = (argc == 4) ? argv[3] : nullptr;

  std::FILE* in = std::fopen(in_path, "r");
  if (!in)
    die("cannot open input file");

  char* line = nullptr;
  std::size_t cap = 0;
  ssize_t len;

  // ---------------- pass 1: header + out-degree counts ----------------
  std::uint64_t n = 0, m_header = 0;
  bool have_header = false;
  std::vector<std::uint64_t> cursor; // degrees now, offsets later
  std::uint64_t m = 0;
  unsigned long long lineno = 0;
  NodeData nodes;

  while ((len = getline(&line, &cap, in)) != -1) {
    ++lineno;
    if (len == 0)
      continue;
    if (line[0] == 'p') {
      char kind[32];
      unsigned long long nn, mm;
      if (std::sscanf(line, "p %31s %llu %llu", kind, &nn, &mm) != 3)
        die("malformed 'p' line", lineno);
      n = nn;
      m_header = mm;
      have_header = true;
      cursor.assign(n, 0);
    } else if (line[0] == 'a') {
      if (!have_header)
        die("'a' line before 'p' line", lineno);
      Arc arc;
      if (!parse_arc(line + 1, arc))
        die("malformed 'a' line", lineno);
      if (arc.u < 1 || arc.u > n || arc.v < 1 || arc.v > n)
        die("node id out of range 1..N", lineno);
      if (arc.w > 0xFFFFFFFFull)
        die("weight does not fit in 32 bits", lineno);
      ++cursor[arc.u - 1];
      ++m;
    }
    // 'c' and anything else: ignored
  }
  if (!have_header)
    die("no 'p' line found");
  if (m != m_header)
    std::fprintf(stderr,
                 "Warning: header says %llu arcs but found %llu; using %llu\n",
                 (unsigned long long)m_header, (unsigned long long)m,
                 (unsigned long long)m);

  std::fprintf(stderr, "nodes=%llu edges=%llu\n", (unsigned long long)n,
               (unsigned long long)m);

  // ---------------- optional separate node-data file ----------------
  if (node_path) {
    std::FILE* nf = std::fopen(node_path, "r");
    if (!nf)
      die("cannot open node data file");
    unsigned long long nline = 0;
    while ((len = getline(&line, &cap, nf)) != -1) {
      ++nline;
      if (len == 0)
        continue;
      if (line[0] == 'n' || line[0] == 'v')
        parse_node_line(line, n, nodes, nline);
      // 'c', 'p' (e.g. "p aux sp co N"), and anything else: ignored
    }
    std::fclose(nf);
    if (nodes.kind == NONE)
      std::fprintf(stderr, "Warning: node data file had no 'n' or 'v' lines\n");
  }

  std::fprintf(stderr, "nodes=%llu edges=%llu node_data=%s\n",
               (unsigned long long)n, (unsigned long long)m,
               nodes.kind == SINGLE ? "single"
               : nodes.kind == PAIR ? "pair(x,y)"
                                    : "none");

  // degrees -> start offsets (exclusive prefix sum)
  {
    std::uint64_t sum = 0;
    for (std::uint64_t i = 0; i < n; ++i) {
      const std::uint64_t d = cursor[i];
      cursor[i] = sum;
      sum += d;
    }
  }

  // ---------------- pass 2: place edges in CSR order ----------------
  std::vector<std::uint32_t> dst(m), wt(m);
  std::rewind(in);
  lineno = 0;
  while ((len = getline(&line, &cap, in)) != -1) {
    ++lineno;
    if (len == 0 || line[0] != 'a')
      continue;
    Arc arc;
    parse_arc(line + 1, arc); // already validated in pass 1
    const std::uint64_t pos = cursor[arc.u - 1]++;
    dst[pos] = static_cast<std::uint32_t>(arc.v - 1);
    wt[pos] = static_cast<std::uint32_t>(arc.w);
  }
  std::free(line);
  std::fclose(in);
  // cursor[i] is now the END offset of node i, which is exactly outIdx.

  // ---------------- write binary file ----------------
  std::FILE* out = std::fopen(out_path, "wb");
  if (!out)
    die("cannot open output file");

  const std::uint64_t header[4] = {1, sizeof(std::uint32_t), n, m};
  bool ok = std::fwrite(header, sizeof(header), 1, out) == 1;
  if (n)
    ok = ok && std::fwrite(cursor.data(), 8, n, out) == n;
  if (m)
    ok = ok && std::fwrite(dst.data(), 4, m, out) == m;
  if (m % 2) {
    const std::uint32_t pad = 0;
    ok = ok && std::fwrite(&pad, 4, 1, out) == 1;
  }
  if (m)
    ok = ok && std::fwrite(wt.data(), 4, m, out) == m;

  unsigned long long expected =
      8ull * (4 + n) + 4ull * (m + (m & 1)) + 4ull * m;

  if (nodes.kind != NONE) {
    const std::uint64_t kind = nodes.kind;
    ok = ok && std::fwrite(&kind, 8, 1, out) == 1;
    if (!nodes.data.empty())
      ok = ok && std::fwrite(nodes.data.data(), 4, nodes.data.size(), out) ==
                     nodes.data.size();
    expected += 8ull + 4ull * nodes.data.size();
  }

  if (std::fclose(out) != 0)
    ok = false;
  if (!ok)
    die("failed while writing output file");

  std::fprintf(stderr, "wrote %s (%llu bytes)\n", out_path, expected);
  return 0;
}