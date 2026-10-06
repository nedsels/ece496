# Input Graphs

Below is a list of input graphs used for this project, where to find them, and how to convert them for use with CSRGraph.

## (1) GER Roads

From repo root, run:
```
wget https://download.geofabrik.de/europe/germany-latest.osm.pbf
./bin/osm_pbf_to_gr germany-latest.osm.pbf ./graphs/inputs/germany-latest.gr
```
