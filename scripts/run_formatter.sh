#!/usr/bin/env bash

set -euo pipefail
IFS=$'\n\t'

function main() {
  # Run formatter
  set +e  # Do not stop if exit status code is non-zero.
  yapf --style='{based_on_style: pep8, column_limit: 100}' -i -r standalone_examples
  set -e
}

# Get the directory containing the script
readonly DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Change to sub-project root folder
pushd ${DIR}/.. > /dev/null

main "$@"
status=$?

popd > /dev/null

exit ${status}