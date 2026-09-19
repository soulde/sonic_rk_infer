# Sonic RK3576 Inference

独立的 RK3576 板端部署包，运行 Sonic low-latency encoder/decoder 的 INT8 RKNN 模型。

## 内容

- `sonic_encoder_int8.rknn`：encoder INT8 模型
- `sonic_decoder_int8.rknn`：decoder INT8 模型
- `sonic_rk3576_infer`：单次推理/基准可执行文件
- `sonic_rk3576_hil_server`：面向 MuJoCo HIL 客户端的 TCP 推理服务
- `inference/`：板端 C++ 源码和 CMake 配置

## 运行

```bash
./sonic_rk3576_infer sonic_encoder_int8.rknn sonic_decoder_int8.rknn
```

HIL 服务默认监听 TCP `18080`：

```bash
./sonic_rk3576_hil_server \
  --encoder sonic_encoder_int8.rknn \
  --decoder sonic_decoder_int8.rknn \
  --port 18080
```

该仓库只包含板端运行所需内容；宿主机上的模型转换、校准、MuJoCo 客户端和测试位于独立的 `sonic_rk_deploy` 仓库。
