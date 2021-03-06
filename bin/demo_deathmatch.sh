#!/bin/bash

function main() {
  python src/demo.py \
   --config=demo_confs/bots_deathmatch_multimaps.json \
   --load=../../trained_agents/deathmatch_512_256-256_stack=4/best_model.zip
}

# Get the directory containing the script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Change to sub-project root folder
pushd ${DIR}/.. > /dev/null

main "$@"
status=$?

popd > /dev/null

exit ${status}