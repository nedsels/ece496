// graph_prep_common.h
// Pieces shared by graph_prep (in memory) and graph_prep_ooc (out of core):
// error handling, compressed input, a fast buffered writer and the edge
// weight hash. Both tools must produce identical weights, so the hash lives
// here only.

#pragma once

#include <algorithm>
#include <charconv>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

[[noreturn]] inline void die(const std::string& msg,
                             unsigned long long line = 0) {
  if (line)
    std::fprintf(stderr, "Error (line %llu): %s\n", line, msg.c_str());
  else
    std::fprintf(stderr, "Error: %s\n", msg.c_str());
  std::exit(1);
}

inline bool ends_with(const std::string& s, const char* suffix) {
  const std::size_t n = std::strlen(suffix);
  return s.size() >= n && s.compare(s.size() - n, n, suffix) == 0;
}

// Opens a plain or compressed file for line reading. .gz goes through pigz
// when available (faster), else gzip.
struct Input {
  std::FILE* f = nullptr;
  bool is_pipe = false;
  explicit Input(const std::string& path) {
    if (path.find('\'') != std::string::npos)
      die("paths must not contain a single quote");
    std::string cmd;
    if (ends_with(path, ".gz"))
      cmd = "{ command -v pigz >/dev/null && pigz -dc '" + path +
            "' || gzip -dc '" + path + "'; }";
    else if (ends_with(path, ".bz2"))
      cmd = "bzip2 -dc '" + path + "'";
    else if (ends_with(path, ".zip"))
      cmd = "unzip -p '" + path + "' '*.mtx'";
    if (!cmd.empty()) {
      f = popen(cmd.c_str(), "r");
      is_pipe = true;
    } else {
      f = std::fopen(path.c_str(), "r");
    }
    if (!f)
      die("cannot open " + path);
  }
  Input(const Input&) = delete;
  Input& operator=(const Input&) = delete;
  ~Input() {
    if (is_pipe) {
      if (pclose(f) != 0)
        die("decompression failed");
    } else {
      std::fclose(f);
    }
  }
};

// Buffered writer with fast integer formatting.
struct Writer {
  std::FILE* f;
  std::string path;
  std::vector<char> buf;
  std::size_t pos = 0;
  explicit Writer(const std::string& p, std::size_t buf_size = 1 << 22)
      : path(p), buf(buf_size) {
    f = std::fopen(p.c_str(), "wb");
    if (!f)
      die("cannot open " + p + " for writing");
  }
  Writer(const Writer&) = delete;
  Writer& operator=(const Writer&) = delete;
  void flush() {
    if (pos && std::fwrite(buf.data(), 1, pos, f) != pos)
      die("write failed: " + path);
    pos = 0;
  }
  void raw(const void* data, std::size_t len) {
    const char* p = static_cast<const char*>(data);
    while (len) {
      if (pos == buf.size())
        flush();
      const std::size_t k = std::min(len, buf.size() - pos);
      std::memcpy(buf.data() + pos, p, k);
      pos += k;
      p += k;
      len -= k;
    }
  }
  template <typename T> void bin(const T& v) { raw(&v, sizeof(T)); }
  void str(const char* s) { raw(s, std::strlen(s)); }
  void ch(char c) {
    if (pos == buf.size())
      flush();
    buf[pos++] = c;
  }
  void num(std::uint64_t v) {
    if (buf.size() - pos < 24)
      flush();
    pos = std::to_chars(buf.data() + pos, buf.data() + buf.size(), v).ptr -
          buf.data();
  }
  ~Writer() {
    flush();
    if (std::fclose(f) != 0)
      die("close failed: " + path);
  }
};

inline std::uint64_t splitmix64(std::uint64_t x) {
  x += 0x9E3779B97F4A7C15ull;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
  return x ^ (x >> 31);
}

// Random weight in [wmin, wmax] for edge {u, v}; the same for (u,v) and (v,u).
inline std::uint32_t random_weight(std::uint32_t u, std::uint32_t v,
                                   std::uint64_t seed, std::uint64_t wmin,
                                   std::uint64_t wmax) {
  const std::uint64_t lo = std::min(u, v), hi = std::max(u, v);
  return static_cast<std::uint32_t>(
      wmin + splitmix64(seed ^ splitmix64((lo << 32) | hi)) % (wmax - wmin + 1));
}
