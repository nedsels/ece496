#pragma once

template <typename Item, typename Priority>
struct Task {
  Item item;
  Priority priority;
};

template <typename Index, typename Priority>
struct VertexInfo {
  Index id;
  // Two interpretations:
  // 1) Encompasses all edges for a vertex and next_edge is atomically incremented by threads
  // 2) Describes only a subset of the edges of a vertex and a single thread is responsible for processing the subset
  // Difference is for 1) only a single entry for a vertex while 2) allows for multiple entries for the same vertex
  // In both cases, each edge is only represented once
  // Uses relative edge number [0, end_edge)
  Index next_edge;
  // Last edge
  Index end_edge;
  Priority priority;
};
