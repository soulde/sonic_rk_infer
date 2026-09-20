#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic \
  inference/src/sonic_rk3576_controller.cpp \
  -Iinference/include \
  -Ideps/unitree_sdk2/include \
  -Ideps/unitree_sdk2/thirdparty/include/ddscxx \
  -Ideps/unitree_sdk2/thirdparty/include \
  -Ldeps/unitree_sdk2/lib/aarch64 \
  -Ldeps/unitree_sdk2/thirdparty/lib/aarch64 \
  -Wl,-rpath,"$PWD/deps/unitree_sdk2/thirdparty/lib/aarch64" \
  -Wl,-rpath,"$PWD/deps/unitree_sdk2/lib/aarch64" \
  -lrknnrt -lunitree_sdk2 -lddscxx -lddsc \
  -lpthread -lrt -ldl \
  -o sonic_rk3576_controller

echo "built ./sonic_rk3576_controller"
