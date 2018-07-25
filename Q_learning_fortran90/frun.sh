#!/bin/bash

make clean
make

nohup ./result > pso.file 2>&1 &