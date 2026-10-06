/*
 csr_graph.cu

 Implements a CSR Graph. Part of the GGC source code.
 Interface derived from LonestarGPU.

 Copyright (C) 2014--2016, The University of Texas at Austin

 See LICENSE.TXT for copyright license.

 Author: Sreepathi Pai <sreepai@ices.utexas.edu>
 */

#include "csr_graph.cuh"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <vector>

template <typename WeightType>
CSRGraph<WeightType>::CSRGraph() {
  init();
}

template <typename WeightType>
unsigned CSRGraph<WeightType>::init() {
  nnodes = 0;
  nedges = 0;
  row_start = nullptr;
  edge_dst = nullptr;
  edge_data = nullptr;
  node_data = nullptr;
  device_graph = false;
  file_name[0] = '\0';
  return 0;
}

template <typename WeightType>
unsigned CSRGraph<WeightType>::allocOnHost() {
  if (nnodes <= 0 || device_graph) {
    return 0;
  }
  if (row_start != nullptr) {
    return 1;
  }

  row_start = static_cast<index_type*>(
      std::calloc(static_cast<std::size_t>(nnodes) + 1, sizeof(index_type)));
  edge_dst = static_cast<index_type*>(
      std::calloc(static_cast<std::size_t>(nedges), sizeof(index_type)));
  edge_data = static_cast<edge_data_type*>(
      std::calloc(static_cast<std::size_t>(nedges), sizeof(edge_data_type)));
  node_data = static_cast<node_data_type*>(
      std::calloc(static_cast<std::size_t>(nnodes), sizeof(node_data_type)));

  if (row_start == nullptr || (nedges > 0 && edge_dst == nullptr) ||
      (nedges > 0 && edge_data == nullptr) || node_data == nullptr) {
    deallocOnHost();
    return 0;
  }
  return 1;
}

template <typename WeightType>
unsigned CSRGraph<WeightType>::allocOnDevice() {
  if (nnodes <= 0 || device_graph || row_start != nullptr) {
    return 0;
  }

  index_type* device_row_start = nullptr;
  index_type* device_edge_dst = nullptr;
  edge_data_type* device_edge_data = nullptr;
  node_data_type* device_node_data = nullptr;

  if (cudaMalloc(&device_row_start, (static_cast<std::size_t>(nnodes) + 1) *
                                        sizeof(index_type)) != cudaSuccess ||
      (nedges > 0 &&
       cudaMalloc(&device_edge_dst, static_cast<std::size_t>(nedges) *
                                        sizeof(index_type)) != cudaSuccess) ||
      (nedges > 0 && cudaMalloc(&device_edge_data,
                                static_cast<std::size_t>(nedges) *
                                    sizeof(edge_data_type)) != cudaSuccess) ||
      cudaMalloc(&device_node_data,
                 static_cast<std::size_t>(nnodes) * sizeof(node_data_type)) !=
          cudaSuccess) {
    cudaFree(device_row_start);
    cudaFree(device_edge_dst);
    cudaFree(device_edge_data);
    cudaFree(device_node_data);
    return 0;
  }

  row_start = device_row_start;
  edge_dst = device_edge_dst;
  edge_data = device_edge_data;
  node_data = device_node_data;
  device_graph = true;
  return 1;
}

template <typename WeightType>
unsigned CSRGraph<WeightType>::deallocOnHost() {
  if (!device_graph) {
    std::free(row_start);
    std::free(edge_dst);
    std::free(edge_data);
    std::free(node_data);
    row_start = nullptr;
    edge_dst = nullptr;
    edge_data = nullptr;
    node_data = nullptr;
  }
  return 0;
}

template <typename WeightType>
unsigned CSRGraph<WeightType>::deallocOnDevice() {
  if (device_graph) {
    cudaFree(row_start);
    cudaFree(edge_dst);
    cudaFree(edge_data);
    cudaFree(node_data);
    row_start = nullptr;
    edge_dst = nullptr;
    edge_data = nullptr;
    node_data = nullptr;
    device_graph = false;
  }
  return 0;
}

template <typename WeightType>
unsigned CSRGraph<WeightType>::dealloc() {
  return device_graph ? deallocOnDevice() : deallocOnHost();
}

template <typename WeightType>
unsigned CSRGraph<WeightType>::copy_to_gpu(CSRGraph& copygraph) const {
  if (device_graph || copygraph.device_graph) {
    return 0;
  }
  copygraph.nnodes = nnodes;
  copygraph.nedges = nedges;
  if (!copygraph.allocOnDevice()) {
    return 0;
  }
  if (cudaMemcpy(copygraph.row_start, row_start,
                 (static_cast<std::size_t>(nnodes) + 1) * sizeof(index_type),
                 cudaMemcpyHostToDevice) != cudaSuccess ||
      (nedges > 0 &&
       (cudaMemcpy(copygraph.edge_dst, edge_dst,
                   static_cast<std::size_t>(nedges) * sizeof(index_type),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(copygraph.edge_data, edge_data,
                   static_cast<std::size_t>(nedges) * sizeof(edge_data_type),
                   cudaMemcpyHostToDevice) != cudaSuccess)) ||
      cudaMemcpy(copygraph.node_data, node_data,
                 static_cast<std::size_t>(nnodes) * sizeof(node_data_type),
                 cudaMemcpyHostToDevice) != cudaSuccess) {
    copygraph.deallocOnDevice();
    return 0;
  }
  return 1;
}

template <typename WeightType>
unsigned CSRGraph<WeightType>::copy_to_cpu(CSRGraph& copygraph) const {
  if (!device_graph) {
    return 0;
  }
  copygraph.nnodes = nnodes;
  copygraph.nedges = nedges;
  if (!copygraph.allocOnHost()) {
    return 0;
  }
  if (cudaMemcpy(copygraph.row_start, row_start,
                 (static_cast<std::size_t>(nnodes) + 1) * sizeof(index_type),
                 cudaMemcpyDeviceToHost) != cudaSuccess ||
      (nedges > 0 &&
       (cudaMemcpy(copygraph.edge_dst, edge_dst,
                   static_cast<std::size_t>(nedges) * sizeof(index_type),
                   cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(copygraph.edge_data, edge_data,
                   static_cast<std::size_t>(nedges) * sizeof(edge_data_type),
                   cudaMemcpyDeviceToHost) != cudaSuccess)) ||
      cudaMemcpy(copygraph.node_data, node_data,
                 static_cast<std::size_t>(nnodes) * sizeof(node_data_type),
                 cudaMemcpyDeviceToHost) != cudaSuccess) {
    copygraph.deallocOnHost();
    return 0;
  }
  return 1;
}

template <typename WeightType>
unsigned CSRGraph<WeightType>::readFromGR(const char* file) {
  if (file == nullptr) {
    return 1;
  }

  std::ifstream input(file, std::ios::binary | std::ios::ate);
  if (!input) {
    return 1;
  }
  const std::streamsize file_size = input.tellg();
  constexpr std::size_t header_size = 4 * sizeof(std::uint64_t);
  if (file_size < static_cast<std::streamsize>(header_size)) {
    return 1;
  }
  input.seekg(0);
  std::vector<unsigned char> bytes(static_cast<std::size_t>(file_size));
  if (!input.read(reinterpret_cast<char*>(bytes.data()), file_size)) {
    return 1;
  }

  const auto read_le64 = [&bytes](std::size_t offset) {
    std::uint64_t value = 0;
    for (unsigned byte = 0; byte < 8; ++byte) {
      value |= static_cast<std::uint64_t>(bytes[offset + byte]) << (byte * 8);
    }
    return value;
  };
  const auto read_le32 = [&bytes](std::size_t offset) {
    std::uint32_t value = 0;
    for (unsigned byte = 0; byte < 4; ++byte) {
      value |= static_cast<std::uint32_t>(bytes[offset + byte]) << (byte * 8);
    }
    return value;
  };

  const std::uint64_t version = read_le64(0);
  const std::uint64_t weight_size = read_le64(8);
  const std::uint64_t node_count = read_le64(16);
  const std::uint64_t edge_count = read_le64(24);
  if (version != 1 || node_count == 0 ||
      node_count >
          static_cast<std::uint64_t>(std::numeric_limits<index_type>::max()) ||
      edge_count >
          static_cast<std::uint64_t>(std::numeric_limits<index_type>::max()) ||
      (weight_size != 0 && weight_size != sizeof(edge_data_type))) {
    return 1;
  }

  const std::size_t row_offset = header_size;
  const std::size_t destination_offset =
      row_offset + static_cast<std::size_t>(node_count) * sizeof(std::uint64_t);
  const std::size_t edge_data_offset =
      destination_offset +
      static_cast<std::size_t>(edge_count + (edge_count & 1)) *
          sizeof(std::uint32_t);
  const std::size_t required_size =
      edge_data_offset +
      (weight_size == 0 ? 0
                        : static_cast<std::size_t>(edge_count) * weight_size);
  if (required_size > bytes.size()) {
    return 1;
  }

  dealloc();
  nnodes = static_cast<index_type>(node_count);
  nedges = static_cast<index_type>(edge_count);
  if (!allocOnHost()) {
    return 1;
  }

  row_start[0] = 0;
  for (index_type node = 0; node < nnodes; ++node) {
    const std::uint64_t row_end =
        read_le64(row_offset + static_cast<std::size_t>(node) * 8);
    if (row_end < static_cast<std::uint64_t>(row_start[node]) ||
        row_end > edge_count) {
      deallocOnHost();
      nnodes = nedges = 0;
      return 1;
    }
    row_start[node + 1] = static_cast<index_type>(row_end);
  }
  if (row_start[nnodes] != nedges) {
    deallocOnHost();
    nnodes = nedges = 0;
    return 1;
  }

  for (index_type edge = 0; edge < nedges; ++edge) {
    const std::uint32_t destination =
        read_le32(destination_offset + static_cast<std::size_t>(edge) * 4);
    if (destination >= node_count) {
      deallocOnHost();
      nnodes = nedges = 0;
      return 1;
    }
    edge_dst[edge] = static_cast<index_type>(destination);
    if (weight_size != 0) {
      if constexpr (std::is_same_v<edge_data_type, float>) {
        const std::uint32_t bits = read_le32(
            edge_data_offset + static_cast<std::size_t>(edge) * weight_size);
        std::memcpy(&edge_data[edge], &bits, sizeof(bits));
      } else {
        using unsigned_weight_type = std::make_unsigned_t<edge_data_type>;
        std::uint64_t bits = 0;
        const std::size_t weight_offset =
            edge_data_offset + static_cast<std::size_t>(edge) * weight_size;
        for (std::size_t byte = 0; byte < sizeof(edge_data_type); ++byte) {
          bits |= static_cast<std::uint64_t>(bytes[weight_offset + byte])
                  << (byte * 8);
        }
        const unsigned_weight_type value =
            static_cast<unsigned_weight_type>(bits);
        std::memcpy(&edge_data[edge], &value, sizeof(value));
      }
    }
  }

  std::snprintf(file_name, sizeof(file_name), "%s", file);
  return 0;
}

template <typename WeightType>
__host__ __device__ bool
CSRGraph<WeightType>::valid_node(index_type node) const {
  return node >= 0 && node < nnodes;
}

template <typename WeightType>
__host__ __device__ bool
CSRGraph<WeightType>::valid_edge(index_type edge) const {
  return edge >= 0 && edge < nedges;
}

template <typename WeightType>
__host__ __device__ typename CSRGraph<WeightType>::index_type
CSRGraph<WeightType>::getOutDegree(index_type src) const {
  return row_start[src + 1] - row_start[src];
}

template <typename WeightType>
__host__ __device__ typename CSRGraph<WeightType>::index_type
CSRGraph<WeightType>::getDestination(index_type src, index_type edge) const {
  return edge_dst[row_start[src] + edge];
}

template <typename WeightType>
__host__ __device__ typename CSRGraph<WeightType>::index_type
CSRGraph<WeightType>::getAbsDestination(index_type edge) const {
  return edge_dst[edge];
}

template <typename WeightType>
__host__ __device__ typename CSRGraph<WeightType>::index_type
CSRGraph<WeightType>::getFirstEdge(index_type src) const {
  return row_start[src];
}

template <typename WeightType>
__host__ __device__ typename CSRGraph<WeightType>::edge_data_type
CSRGraph<WeightType>::getWeight(index_type src, index_type edge) const {
  return edge_data[row_start[src] + edge];
}

template <typename WeightType>
__host__ __device__ typename CSRGraph<WeightType>::edge_data_type
CSRGraph<WeightType>::getAbsWeight(index_type edge) const {
  return edge_data[edge];
}

template struct CSRGraph<std::int32_t>;
template struct CSRGraph<std::uint32_t>;
template struct CSRGraph<float>;
