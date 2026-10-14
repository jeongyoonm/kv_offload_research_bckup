# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
cmake_minimum_required(VERSION 3.18)
project(predictor_fast_forest LANGUAGES CXX)
add_library(predictor_forest_walk SHARED forest_walk.cpp)
target_compile_features(predictor_forest_walk PRIVATE cxx_std_17)
target_compile_options(predictor_forest_walk PRIVATE -Wall -Wextra -Werror)
