#!/bin/bash
mkdir -p bin
cd bin 
echo -e "\e[34m[-]\e[0m running cmake..."
cmake ../ -DCMAKE_BUILD_TYPE=Debug
# cmake ../ 
echo -e "\e[34m[-]\e[0m running make..."
make

echo -e "\e[34m[v]\e[0m Done"