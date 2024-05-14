#!/bin/bash

# Create directories for implementations
mkdir -p Implementations
cd Implementations

# Create directories for Zero Layer Transformers
mkdir -p Zero_Layer_Transformers


# Create directories for One Layer Transformers
mkdir -p One_Layer_Transformers


# Create directories for Two Layer Transformers
mkdir -p Two_Layer_Transformers




# Create directory for Experiment Goals
mkdir -p Experiment_Goals
cd Experiment_Goals

# Create file for Experiment Goal 1
touch "Experiment_Goal_1.txt"
echo "Goal 1: Understand the architecture in detail" >> "Experiment_Goal_1.txt"
echo "Progress: Incomplete" >> "Experiment_Goal_1.txt"

# Create file for Experiment Goal 2
touch "Experiment_Goal_2.txt"
echo "Goal 2: Understand how directions in activation space relate to emergent qualities" >> "Experiment_Goal_2.txt"
echo "Progress: Incomplete" >> "Experiment_Goal_2.txt"

# Create file for Experiment Goal 3
touch "Experiment_Goal_3.txt"
echo "Goal 3: Test unitary rotation layer hypothesis" >> "Experiment_Goal_3.txt"
echo "Progress: Incomplete" >> "Experiment_Goal_3.txt"

# Go back to the Alignment Research Directory
cd ../
