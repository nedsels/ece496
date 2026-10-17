// dimacs2gr.cpp
// Convert a DIMACS text graph ("p sp N M" / "a u v [w]", 1-based) into the
// Galois binary .gr format read by CSRGraph::readFromGR:
//   u64 version(=1), u64 sizeof(edge data)(=4), u64 numNodes, u64 numEdges
//   u64 outIdx[numNodes]     (cumulative end offset of each node's edges)
//   u32 dst[numEdges]        (0-based)
//   u32 pad (only if numEdges is odd)
//   u32 weight[numEdges]
//
// Two passes over the text file keep memory low:
//   pass 1 counts out-degrees, pass 2 places each edge directly in CSR order.
// Edges that have no weight field get weight 1.
//
// Build: g++ -O2 -std=c++17 -o dimacs2gr dimacs2gr.cpp
// Usage: ./dimacs2gr input.gr output.bin.gr

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

int main(int argc, char** argv) {
  if (argc != 3) {
    std::fprintf(stderr, "Usage: %s <input_dimacs.gr> <output_binary.gr>\n",
                 argv[0]);
    return 1;
  }
  if (!is_little_endian())
    die("big-endian hosts are not supported");

  const char* in_path = argv[1];
  const char* out_path = argv[2];

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
  if (std::fclose(out) != 0)
    ok = false;
  if (!ok)
    die("failed while writing output file");

  const unsigned long long expected =
      8ull * (4 + n) + 4ull * (m + (m & 1)) + 4ull * m;
  std::fprintf(stderr, "wrote %s (%llu bytes)\n", out_path, expected);
  return 0;
}