#!/bin/bash

find include src unit_tests -type f -iname *.cpp -o -iname *.hpp -o -iname *.cu -o -iname *.cuh | xargs clang-format -i
