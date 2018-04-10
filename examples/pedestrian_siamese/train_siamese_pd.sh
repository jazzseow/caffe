#!/usr/bin/env sh
set -e

TOOLS=./build/tools

$TOOLS/caffe train --solver=examples/pedestrian_siamese/siamese_pd_solver.prototxt \
	-snapshot examples/pedestrian_siamese/siameseDW/siameseDW_iter_113338.solverstate
