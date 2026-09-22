# sonic_rk_infer

RK3576-side Sonic low-latency controller.  The board subscribes to the
latest `rt/lowstate`, runs the 50 Hz encoder/decoder policy with RKNN, and
publishes `rt/lowcmd` at 500 Hz.  The DDS callback is latest-only; it does
not queue old observations.

The controller has the same three states as the reference implementation:

`INIT` ramps all 29 body motors to the official default pose, then enters
`WAIT_FOR_CONTROL`; `--auto-start` starts policy control and `--auto-play`
enables the loaded motion reference.  Without either flag it remains in the
wait state after initialization.

## Build

The board needs the RKNN runtime and the copied Unitree SDK2 AArch64
dependencies under `deps/unitree_sdk2`.  Build with:

```bash
./build_controller.sh
```

## MuJoCo / DDS test

On the board:

```bash
DDS_DOMAIN=24 DDS_INTERFACE=eth0 ./run_controller.sh --auto-start --auto-play
```

The host simulator must use the same DDS domain and the host NIC connected to
the board.  The RKNN model inputs are float32 tensors with dimensions 1247
and 994; the default deployment models are the validated Float16 models
`sonic_encoder_float.rknn` and `sonic_decoder_float.rknn`.  The INT8 candidates
remain available as `sonic_encoder_int8.rknn` and `sonic_decoder_int8.rknn` and
can be selected with `ENCODER_MODEL` and `DECODER_MODEL`.

This controller intentionally has no MuJoCo or host-project dependency.
