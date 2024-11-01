#!/usr/bin/bash

mkdir tmp || exit
cd tmp || exit
curl https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh --output anaconda.sh || exit
sha256sum anaconda.sh || exit
bash anaconda.sh -b -p ./../anaconda3 || exit
cd ..
rm -r tmp
