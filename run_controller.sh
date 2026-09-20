#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
export LD_LIBRARY_PATH="/lib:/usr/lib:$PWD/deps/unitree_sdk2/thirdparty/lib/aarch64:$PWD/deps/unitree_sdk2/lib/aarch64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

exec stdbuf -oL -eL ./sonic_rk3576_controller \
  --encoder "${ENCODER_MODEL:-sonic_encoder_int8.rknn}" \
  --decoder "${DECODER_MODEL:-sonic_decoder_int8.rknn}" \
  --motion "${MOTION_DIR:-reference/dance_in_da_party_001__A464_M}" \
  --domain "${DDS_DOMAIN:-0}" \
  --interface "${DDS_INTERFACE:-eth0}" \
  "${@}"
