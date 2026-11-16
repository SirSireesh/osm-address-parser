#!/bin/sh
# both stages end to end: pbf → buckets → training jsonl.
# usage: run.sh -i <pbf> -o <bucketdir> [dataset options...]
set -eu

pbf=
dir=

while [ $# -gt 0 ]; do
	case $1 in
	-i) pbf=$2; shift 2 ;;
	-o) dir=$2; shift 2 ;;
	*) break ;;
	esac
done

[ -n "$pbf" ] && [ -n "$dir" ] || {
	echo "usage: run.sh -i <pbf> -o <bucketdir> [dataset options...]" >&2
	exit 1
}

bin/osm_address_extractor -i "$pbf" -o "$dir"
bin/osm_address_extractor-dataset -i "$dir" "$@"
