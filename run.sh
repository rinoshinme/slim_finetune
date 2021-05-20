#!/bin/bash

CUDA_VISIBLE_DEVICES='2' nohup python main.py > log.txt 2>&1 &
