#!/bin/bash
#
# deploy.sh — Upload the built .vsix package to the JFrog Artifactory repository.
#
# This script reads credentials (USERNAME, PASSWORD) from environment
# variables and uploads the .vsix archive produced by build.sh to the
# configured Artifactory server.
#
# Prerequisites:
#   - build.sh must have been run successfully first.
#   - The environment variables USERNAME and PASSWORD must be set.
#
# Usage:
#   USERNAME=<user> PASSWORD=<pass> ./scripts/deploy.sh
#
# Examples:
#   USERNAME=deploy PASSWORD=s3cret ./scripts/deploy.sh
#

set -euo pipefail

ROOT_DIR="$(readlink -f "$(dirname "$BASH_SOURCE")/..")"
cd "${ROOT_DIR}"

# Read version from package.json
PACKAGE_VERSION="$(node -e "console.log(require('./package.json').version)")"
PACKAGE_NAME="tinycoder"
PACKAGE_FULLNAME="${PACKAGE_NAME}-${PACKAGE_VERSION}.vsix"
PACKAGE_PATH="${ROOT_DIR}/${PACKAGE_FULLNAME}"

if [ ! -f "${PACKAGE_PATH}" ]; then
    echo "ERROR: Package not found at ${PACKAGE_PATH}"
    echo "Run ./scripts/build.sh first to produce the .vsix file."
    exit 1
fi

# Artifactory upload URL
URL="https://mgorshkov.jfrog.io/artifactory/default-generic-local/tinycoder/${PACKAGE_FULLNAME}"

echo "Uploading ${PACKAGE_FULLNAME} to Artifactory..."
echo "  URL: ${URL}"

# Upload the package using HTTP PUT with basic auth
curl -T "${PACKAGE_PATH}" -u "${USERNAME}:${PASSWORD}" "${URL}"

echo ""
echo "=== Deploy complete ==="
