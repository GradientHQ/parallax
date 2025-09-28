PYTHON_VERSION=$(python3 -c 'import sys; print("{}.{}".format(sys.version_info.major, sys.version_info.minor))')
REQUIRED_VERSION="3.11"

version_ge() {
    # returns 0 if $1 >= $2
    [ "$(printf '%s\n' "$2" "$1" | sort -V | head -n1)" = "$2" ]
}

if ! version_ge "$PYTHON_VERSION" "$REQUIRED_VERSION"; then
    echo "Error: Python 3.11 or higher is required. Current version is $PYTHON_VERSION."
    exit 1
fi
