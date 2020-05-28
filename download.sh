#/bin/sh

# TODO fix for any path

home="$(pwd)"
cd data
mkdir _domains

git clone https://github.com/tb0hdan/domains.git
cd domains
./unpack.sh

find . -type f -name \*.txt -exec cp \{\} "$home/data/_domains" \;
cd "$home/data/_domains"
ls | xargs -I{ cat { > "$home/data/domains.txt"
