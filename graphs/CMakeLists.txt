add_executable(osm_pbf_to_gr converters/osm_pbf_to_gr.cpp)
add_executable(dimacs_to_gr converters/dimacs_to_gr.cpp)
set_target_properties(osm_pbf_to_gr PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin"
)
set_target_properties(dimacs_to_gr PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin"
)