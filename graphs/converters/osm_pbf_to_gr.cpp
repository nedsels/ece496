#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

// ---------------------------------------------------------------------------
// Minimal inflate (RFC 1951), based on the structure of zlib's "puff".
// ---------------------------------------------------------------------------
class Inflater {
public:
  Inflater(const std::uint8_t* in, std::size_t len,
           std::vector<std::uint8_t>& out)
      : in_(in), len_(len), out_(out) {}

  void run() {
    int last;
    do {
      last = bits(1);
      const int type = bits(2);
      if (type == 0)
        stored();
      else if (type == 1)
        fixed();
      else if (type == 2)
        dynamic();
      else
        throw std::runtime_error("inflate: bad block type");
    } while (!last);
  }

private:
  struct Huffman {
    std::uint16_t count[16];
    std::uint16_t symbol[288];
  };

  int bits(int need) {
    while (bitcnt_ < need) {
      if (pos_ >= len_)
        throw std::runtime_error("inflate: out of input");
      bitbuf_ |= static_cast<std::uint32_t>(in_[pos_++]) << bitcnt_;
      bitcnt_ += 8;
    }
    const int val = static_cast<int>(bitbuf_ & ((1u << need) - 1));
    bitbuf_ >>= need;
    bitcnt_ -= need;
    return val;
  }

  static void build(Huffman& h, const std::uint16_t* length, int n) {
    for (int i = 0; i < 16; ++i)
      h.count[i] = 0;
    for (int s = 0; s < n; ++s)
      h.count[length[s]]++;
    std::uint16_t offs[16];
    offs[1] = 0;
    for (int l = 1; l < 15; ++l)
      offs[l + 1] = offs[l] + h.count[l];
    for (int s = 0; s < n; ++s)
      if (length[s] != 0)
        h.symbol[offs[length[s]]++] = static_cast<std::uint16_t>(s);
  }

  int decode(const Huffman& h) {
    int code = 0, first = 0, index = 0;
    for (int len = 1; len <= 15; ++len) {
      code |= bits(1);
      const int count = h.count[len];
      if (code - count < first)
        return h.symbol[index + (code - first)];
      index += count;
      first += count;
      first <<= 1;
      code <<= 1;
    }
    throw std::runtime_error("inflate: bad huffman code");
  }

  void stored() {
    bitbuf_ = 0;
    bitcnt_ = 0;
    if (pos_ + 4 > len_)
      throw std::runtime_error("inflate: truncated");
    const unsigned n = in_[pos_] | (in_[pos_ + 1] << 8);
    const unsigned nn = in_[pos_ + 2] | (in_[pos_ + 3] << 8);
    pos_ += 4;
    if (n != (~nn & 0xffffu))
      throw std::runtime_error("inflate: bad stored block");
    if (pos_ + n > len_)
      throw std::runtime_error("inflate: truncated");
    out_.insert(out_.end(), in_ + pos_, in_ + pos_ + n);
    pos_ += n;
  }

  void codes(const Huffman& lencode, const Huffman& distcode) {
    static const std::uint16_t lens[29] = {
        3,  4,  5,  6,  7,  8,  9,  10, 11,  13,  15,  17,  19,  23, 27,
        31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258};
    static const std::uint16_t lext[29] = {0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
                                           1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
                                           4, 4, 4, 4, 5, 5, 5, 5, 0};
    static const std::uint16_t dists[30] = {
        1,    2,    3,    4,    5,    7,    9,    13,    17,    25,
        33,   49,   65,   97,   129,  193,  257,  385,   513,   769,
        1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577};
    static const std::uint16_t dext[30] = {
        0, 0, 0, 0, 1, 1, 2, 2,  3,  3,  4,  4,  5,  5,  6,
        6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13};
    for (;;) {
      int sym = decode(lencode);
      if (sym < 256) {
        out_.push_back(static_cast<std::uint8_t>(sym));
      } else if (sym == 256) {
        return;
      } else {
        sym -= 257;
        if (sym >= 29)
          throw std::runtime_error("inflate: bad length symbol");
        const int length = lens[sym] + bits(lext[sym]);
        const int ds = decode(distcode);
        if (ds >= 30)
          throw std::runtime_error("inflate: bad distance symbol");
        const std::size_t dist = dists[ds] + bits(dext[ds]);
        if (dist > out_.size())
          throw std::runtime_error("inflate: distance too far");
        const std::size_t from = out_.size() - dist;
        for (int i = 0; i < length; ++i) {
          const std::uint8_t b = out_[from + i];
          out_.push_back(b);
        }
      }
    }
  }

  void fixed() {
    std::uint16_t lengths[288];
    int s = 0;
    for (; s < 144; ++s)
      lengths[s] = 8;
    for (; s < 256; ++s)
      lengths[s] = 9;
    for (; s < 280; ++s)
      lengths[s] = 7;
    for (; s < 288; ++s)
      lengths[s] = 8;
    Huffman lencode, distcode;
    build(lencode, lengths, 288);
    for (s = 0; s < 30; ++s)
      lengths[s] = 5;
    build(distcode, lengths, 30);
    codes(lencode, distcode);
  }

  void dynamic() {
    static const std::uint8_t order[19] = {16, 17, 18, 0, 8,  7, 9,  6, 10, 5,
                                           11, 4,  12, 3, 13, 2, 14, 1, 15};
    const int nlen = bits(5) + 257;
    const int ndist = bits(5) + 1;
    const int ncode = bits(4) + 4;
    if (nlen > 286 || ndist > 30)
      throw std::runtime_error("inflate: bad counts");
    std::uint16_t lengths[320];
    int idx = 0;
    for (; idx < ncode; ++idx)
      lengths[order[idx]] = static_cast<std::uint16_t>(bits(3));
    for (; idx < 19; ++idx)
      lengths[order[idx]] = 0;
    Huffman lencode, distcode;
    build(lencode, lengths, 19);
    idx = 0;
    while (idx < nlen + ndist) {
      int sym = decode(lencode);
      if (sym < 16) {
        lengths[idx++] = static_cast<std::uint16_t>(sym);
      } else {
        int len = 0, rep;
        if (sym == 16) {
          if (idx == 0)
            throw std::runtime_error("inflate: bad repeat");
          len = lengths[idx - 1];
          rep = 3 + bits(2);
        } else if (sym == 17) {
          rep = 3 + bits(3);
        } else {
          rep = 11 + bits(7);
        }
        if (idx + rep > nlen + ndist)
          throw std::runtime_error("inflate: too many lengths");
        while (rep--)
          lengths[idx++] = static_cast<std::uint16_t>(len);
      }
    }
    build(lencode, lengths, nlen);
    build(distcode, lengths + nlen, ndist);
    codes(lencode, distcode);
  }

  const std::uint8_t* in_;
  std::size_t len_;
  std::size_t pos_ = 0;
  std::uint32_t bitbuf_ = 0;
  int bitcnt_ = 0;
  std::vector<std::uint8_t>& out_;
};

// zlib stream = 2-byte header + deflate data + 4-byte adler32 (ignored here).
void zlib_decompress(const std::uint8_t* in, std::size_t len,
                     std::size_t expected, std::vector<std::uint8_t>& out) {
  if (len < 2 || (in[1] & 0x20))
    throw std::runtime_error("bad zlib header");
  out.clear();
  out.reserve(expected);
  Inflater(in + 2, len - 2, out).run();
}

// ---------------------------------------------------------------------------
// Minimal protobuf reader
// ---------------------------------------------------------------------------
struct Span {
  const std::uint8_t* p = nullptr;
  const std::uint8_t* end = nullptr;
  std::size_t size() const { return static_cast<std::size_t>(end - p); }
};

struct Field {
  std::uint32_t num = 0;
  int wire = 0;
  std::uint64_t value = 0; // wire type 0
  Span data;               // wire type 2
};

std::uint64_t read_varint(Span& s) {
  std::uint64_t v = 0;
  int shift = 0;
  for (;;) {
    if (s.p >= s.end)
      throw std::runtime_error("protobuf: truncated varint");
    const std::uint8_t b = *s.p++;
    v |= static_cast<std::uint64_t>(b & 0x7f) << shift;
    if (!(b & 0x80))
      return v;
    shift += 7;
    if (shift > 63)
      throw std::runtime_error("protobuf: varint too long");
  }
}

bool next_field(Span& s, Field& f) {
  if (s.p >= s.end)
    return false;
  const std::uint64_t tag = read_varint(s);
  f.num = static_cast<std::uint32_t>(tag >> 3);
  f.wire = static_cast<int>(tag & 7);
  switch (f.wire) {
  case 0:
    f.value = read_varint(s);
    break;
  case 1:
    if (s.size() < 8)
      throw std::runtime_error("protobuf: truncated");
    s.p += 8;
    break;
  case 2: {
    const std::uint64_t n = read_varint(s);
    if (n > s.size())
      throw std::runtime_error("protobuf: truncated");
    f.data = {s.p, s.p + n};
    s.p += n;
    break;
  }
  case 5:
    if (s.size() < 4)
      throw std::runtime_error("protobuf: truncated");
    s.p += 4;
    break;
  default:
    throw std::runtime_error("protobuf: unsupported wire type");
  }
  return true;
}

// ---------------------------------------------------------------------------
// Graph building (same rules as the libosmium version: any way with a
// "highway" tag, one directed edge per consecutive node pair)
// ---------------------------------------------------------------------------
struct Graph {
  std::unordered_map<std::int64_t, std::uint32_t> node_to_idx;
  std::vector<std::pair<std::uint32_t, std::uint32_t>> edges;

  std::uint32_t index_of(std::int64_t id) {
    auto it = node_to_idx.find(id);
    if (it != node_to_idx.end())
      return it->second;
    const auto idx = static_cast<std::uint32_t>(node_to_idx.size());
    node_to_idx.emplace(id, idx);
    return idx;
  }
};

void process_way(Span way, std::uint32_t highway_key, Graph& g,
                 std::vector<std::uint32_t>& keys,
                 std::vector<std::int64_t>& refs) {
  keys.clear();
  refs.clear();
  Field f;
  while (next_field(way, f)) {
    if (f.wire != 2)
      continue;
    if (f.num == 2) { // keys (packed uint32)
      Span s = f.data;
      while (s.p < s.end)
        keys.push_back(static_cast<std::uint32_t>(read_varint(s)));
    } else if (f.num == 8) { // refs (packed sint64, delta coded)
      Span s = f.data;
      std::int64_t acc = 0;
      while (s.p < s.end) {
        const std::uint64_t z = read_varint(s);
        acc += static_cast<std::int64_t>(z >> 1) ^
               -static_cast<std::int64_t>(z & 1);
        refs.push_back(acc);
      }
    }
  }
  bool is_highway = false;
  for (const auto k : keys) {
    if (k == highway_key) {
      is_highway = true;
      break;
    }
  }
  if (!is_highway)
    return;
  for (std::size_t i = 0; i + 1 < refs.size(); ++i) {
    const std::uint32_t u = g.index_of(refs[i]);
    const std::uint32_t v = g.index_of(refs[i + 1]);
    g.edges.emplace_back(u, v);
  }
}

void process_block(Span block, Graph& g) {
  std::vector<std::string_view> strings;
  std::vector<Span> groups;
  Field f;
  while (next_field(block, f)) {
    if (f.wire != 2)
      continue;
    if (f.num == 1) { // StringTable
      Span st = f.data;
      Field s;
      while (next_field(st, s)) {
        if (s.num == 1 && s.wire == 2)
          strings.emplace_back(reinterpret_cast<const char*>(s.data.p),
                               s.data.size());
      }
    } else if (f.num == 2) { // PrimitiveGroup
      groups.push_back(f.data);
    }
  }
  std::uint32_t highway_key = 0;
  bool found = false;
  for (std::uint32_t i = 0; i < strings.size(); ++i) {
    if (strings[i] == "highway") {
      highway_key = i;
      found = true;
      break;
    }
  }
  if (!found)
    return;

  std::vector<std::uint32_t> keys;
  std::vector<std::int64_t> refs;
  for (Span group : groups) {
    Field gf;
    while (next_field(group, gf)) {
      if (gf.num == 3 && gf.wire == 2)
        process_way(gf.data, highway_key, g, keys, refs);
    }
  }
}

// ---------------------------------------------------------------------------
// Buffered little-endian output
// ---------------------------------------------------------------------------
class LeWriter {
public:
  explicit LeWriter(std::ostream& out) : out_(out) {
    buf_.reserve(kFlush + 16);
  }
  ~LeWriter() { flush(); }
  void u32(std::uint32_t v) {
    for (unsigned i = 0; i < 4; ++i)
      buf_.push_back(static_cast<char>(v >> (i * 8)));
    maybe_flush();
  }
  void u64(std::uint64_t v) {
    for (unsigned i = 0; i < 8; ++i)
      buf_.push_back(static_cast<char>(v >> (i * 8)));
    maybe_flush();
  }
  void flush() {
    if (!buf_.empty())
      out_.write(buf_.data(), static_cast<std::streamsize>(buf_.size()));
    buf_.clear();
  }

private:
  static constexpr std::size_t kFlush = 1 << 20;
  void maybe_flush() {
    if (buf_.size() >= kFlush)
      flush();
  }
  std::ostream& out_;
  std::vector<char> buf_;
};

void read_pbf(const std::string& path, Graph& g) {
  std::ifstream in(path, std::ios::binary);
  if (!in)
    throw std::runtime_error("could not open " + path);

  std::vector<std::uint8_t> header, blob, inflated;
  constexpr std::uint32_t kMaxHeader = 64 * 1024;
  constexpr std::uint32_t kMaxBlob = 64 * 1024 * 1024;

  for (;;) {
    std::uint8_t lenbuf[4];
    if (!in.read(reinterpret_cast<char*>(lenbuf), 4))
      break; // clean EOF
    const std::uint32_t header_len =
        (std::uint32_t(lenbuf[0]) << 24) | (std::uint32_t(lenbuf[1]) << 16) |
        (std::uint32_t(lenbuf[2]) << 8) | std::uint32_t(lenbuf[3]);
    if (header_len > kMaxHeader)
      throw std::runtime_error("BlobHeader too large");
    header.resize(header_len);
    if (!in.read(reinterpret_cast<char*>(header.data()), header_len))
      throw std::runtime_error("truncated BlobHeader");

    std::string type;
    std::uint32_t datasize = 0;
    Span hs{header.data(), header.data() + header.size()};
    Field f;
    while (next_field(hs, f)) {
      if (f.num == 1 && f.wire == 2)
        type.assign(reinterpret_cast<const char*>(f.data.p), f.data.size());
      else if (f.num == 3 && f.wire == 0)
        datasize = static_cast<std::uint32_t>(f.value);
    }
    if (datasize > kMaxBlob)
      throw std::runtime_error("Blob too large");
    blob.resize(datasize);
    if (!in.read(reinterpret_cast<char*>(blob.data()), datasize))
      throw std::runtime_error("truncated Blob");
    if (type != "OSMData")
      continue; // skip OSMHeader and unknown blocks

    Span raw, zlib;
    std::uint64_t raw_size = 0;
    bool has_raw = false, has_zlib = false;
    Span bs{blob.data(), blob.data() + blob.size()};
    while (next_field(bs, f)) {
      if (f.num == 1 && f.wire == 2) {
        raw = f.data;
        has_raw = true;
      } else if (f.num == 2 && f.wire == 0)
        raw_size = f.value;
      else if (f.num == 3 && f.wire == 2) {
        zlib = f.data;
        has_zlib = true;
      } else if (f.num >= 4 && f.num <= 7 && f.wire == 2)
        throw std::runtime_error(
            "unsupported blob compression (only raw and zlib)");
    }
    if (has_zlib) {
      zlib_decompress(zlib.p, zlib.size(), static_cast<std::size_t>(raw_size),
                      inflated);
      process_block(Span{inflated.data(), inflated.data() + inflated.size()},
                    g);
    } else if (has_raw) {
      process_block(raw, g);
    }
  }
}

} // namespace

int main(int argc, char* argv[]) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <input.osm.pbf> [output_path.gr]\n";
    return 1;
  }
  const std::string input_pbf = argv[1];
  const std::string output_gr = argc >= 3 ? argv[2] : "output.gr";

  Graph g;
  try {
    std::cout << "Reading input map file: " << input_pbf << "...\n";
    read_pbf(input_pbf, g);
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }

  std::cout << "Writing graph to custom path: " << output_gr << "...\n";
  std::ofstream gr_file(output_gr, std::ios::binary);
  if (!gr_file.is_open()) {
    std::cerr << "Error: Could not open or create output file at " << output_gr
              << "\n";
    return 1;
  }

  const auto node_count = static_cast<std::uint32_t>(g.node_to_idx.size());
  const std::uint64_t edge_count = g.edges.size();

  // CSR via stable counting sort (keeps per-row edge order identical to the old
  // tool).
  std::vector<std::uint64_t> row_end(node_count, 0);
  for (const auto& e : g.edges)
    row_end[e.first]++;
  for (std::uint32_t i = 1; i < node_count; ++i)
    row_end[i] += row_end[i - 1];
  std::vector<std::uint32_t> dest(edge_count);
  {
    std::vector<std::uint64_t> pos(node_count, 0);
    for (std::uint32_t i = 1; i < node_count; ++i)
      pos[i] = row_end[i - 1];
    for (const auto& e : g.edges)
      dest[pos[e.first]++] = e.second;
  }

  {
    LeWriter w(gr_file);
    w.u64(1);                     // version
    w.u64(sizeof(std::uint32_t)); // edge data size
    w.u64(node_count);
    w.u64(edge_count);
    for (const auto r : row_end)
      w.u64(r);
    for (const auto d : dest)
      w.u32(d);
    if (edge_count & 1)
      w.u32(0); // pad to 8-byte alignment
    for (std::uint64_t i = 0; i < edge_count; ++i)
      w.u32(1);
  }

  std::cout << "Done! Saved " << edge_count << " edges successfully.\n";
  return 0;
}