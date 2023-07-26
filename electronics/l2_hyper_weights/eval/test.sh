#!/bin/bash

echo "=======================train===================="
grep "Epoch [0-9]*:" logs/*-worker-*.log | cut -d":" -f2- | sort -k5 -n | python eval/train.py
echo "=======================test====================="
grep "test==" logs/*-worker-*.log | cut -d":" -f2- | sort -k5 -n | python eval/test.py
