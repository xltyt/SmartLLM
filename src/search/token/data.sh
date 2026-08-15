#!/bin/bash

# apt install zstd

mkdir -p /data/token_batch_data/
cp custom.txt /data/token_batch_data/

cd /data/token_batch_data/

# https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T
declare -A dir_count
wget 'https://data.together.xyz/redpajama-data-1T/v1.0.0/urls.txt'
while read line; do
  dload_loc=${line#https://data.together.xyz/redpajama-data-1T/v1.0.0/}
  dir=$(dirname "$dload_loc")
  if [ "${dir_count[$dir]:-0}" -ge 2 ]; then
    continue
  fi
  mkdir -p $dir
  if [ ! -f "$dir/$(basename $dload_loc)" ]; then
    wget "$line" -O "$dir/$(basename $dload_loc)"
  fi
  dir_count[$dir]=$(( ${dir_count[$dir]:-0} + 1 ))
done < urls.txt
rm -f urls.txt
for line in $(find -name '*.zst'); do
  zstd -d --rm $line
done

wget https://www.unicode.org/Public/UCD/latest/emoji/emoji-test.txt

python data.py

# vim: set expandtab ts=4 sw=4 sts=4:
