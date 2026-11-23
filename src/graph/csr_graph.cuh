/*
 csr_graph.h

 Implements a CSR Graph. Part of the GGC source code.
 Interface derived from LonestarGPU.

 Copyright (C) 2014--2016, The University of Texas at Austin

 See LICENSE.TXT for copyright license.

 Author: Sreepathi Pai <sreepai@ices.utexas.edu>
 */

#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <type_traits>
#include <utility>

// WeightType: Data type (int, uint, float) for edge weight
// NodeDataType: Data type for value stored on node (i.e. int for pagerank, {int, int} for A*)
template <typename WeightType, typename NodeDataType>
struct CSRGraph {
  using index_type = std::int32_t;
  using edge_data_type = WeightType;
  using node_data_type = NodeDataType;

  CSRGraph();

  unsigned init();
  unsigned allocOnHost();
  unsigned allocOnDevice();
  unsigned deallocOnHost();
  unsigned deallocOnDevice();
  unsigned dealloc();
  unsigned copy_to_gpu(CSRGraph& copygraph) const;
  unsigned copy_to_cpu(CSRGraph& copygraph) const;
  unsigned readFromGR(const char* file);

  __host__ __device__ bool valid_node(index_type node) const;
  __host__ __device__ bool valid_edge(index_type edge) const;
  __host__ __device__ index_type getOutDegree(index_type src) const;
  __host__ __device__ index_type getDestination(index_type src,
                                                index_type edge) const;
  __host__ __device__ index_type getAbsDestination(index_type edge) const;
  __host__ __device__ index_type getFirstEdge(index_type src) const;
  __host__ __device__ edge_data_type getWeight(index_type src,
                                               index_type edge) const;
  __host__ __device__ edge_data_type getAbsWeight(index_type edge) const;
  __host__ __device__ node_data_type getNodeData(index_type node) const;

  index_type nnodes;
  index_type nedges;
  index_type* row_start;
  index_type* edge_dst;
  edge_data_type* edge_data; /* Data stored for each edge i.e. weigth */ 
  node_data_type* node_data; /* Data stored for each node i.e. value or location */
  bool device_graph;
  char file_name[256];
};

extern template struct CSRGraph<std::int32_t, std::int32_t>;
extern template struct CSRGraph<std::uint32_t, std::int32_t>;
extern template struct CSRGraph<float, std::int32_t>;

extern template struct CSRGraph<std::int32_t, std::pair<std::int32_t, std::int32_t>>;
extern template struct CSRGraph<std::uint32_t, std::pair<std::int32_t, std::int32_t>>;
extern template struct CSRGraph<float, std::pair<std::int32_t, std::int32_t>>;
