#pragma once

template <typename Item, typename Priority>
struct Task {
  Item item;
  Priority priority;
};

template <typename Index, typename Priority>
struct VertexInfo {
  Index id;
  Index degree;
  Priority priority;
};
