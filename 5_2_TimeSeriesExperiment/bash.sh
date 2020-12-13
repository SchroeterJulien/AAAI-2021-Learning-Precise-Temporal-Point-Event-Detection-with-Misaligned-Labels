#!/bin/bash --login

for noise in 0 1 2 3 4
do
	for run in 0 1 2 3 4 5 6 7 8 9 
	do
		for fold in 0 1 2 3 4 5
		do
		# Train
		python3 dataProcessing.py $noise $fold 
		done
	done
done


