add_executable(osm_pbf_to_gr converters/osm_pbf_to_gr.cpp)
add_executable(dimacs_to_gr converters/dimacs_to_gr.cpp)
set_target_properties(osm_pbf_to_gr PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin"
)
set_target_properties(dimacs_to_gr PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin"
)

add_executable(graph_prep converters/graph_prep.cpp)
target_compile_options(graph_prep PRIVATE -O3)
set_target_properties(graph_prep PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin"
)

add_executable(graph_prep_ooc converters/graph_prep_ooc.cpp)
target_compile_options(graph_prep_ooc PRIVATE -O3)
set_target_properties(graph_prep_ooc PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin"
)
