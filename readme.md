# hls4ml on Alveo U50 (HLS C/C++ Kernel)
## Vitis version 2019.2
## Activate the tool 
```bash
source /opt/Xilinx/Vitis/2019.2/settings64.sh # Vitis
source /opt/xilinx/xrt/setup.sh # Vitis XRT
```
## Compile project
```bash
make cleanall # clean all of the related files
make check TARGET=sw_emu DEVICE=xilinx_u50_xdma_201920_1 all  # software emulation
make check TARGET=hw_emu DEVICE=xilinx_u50_xdma_201920_1 all  # hardware emulation
make TARGET=hw DEVICE=xilinx_u50_xdma_201920_1 all # build
```
## Run project
```bash
XCL_EMULATION_MODE=sw_emu ./host ./build_dir.sw_emu.xilinx_u50_xdma_201920_1/alveo_hls4ml.xclbin  # software emulation
XCL_EMULATION_MODE=hw_emu ./host ./build_dir.hw_emu.xilinx_u50_xdma_201920_1/alveo_hls4ml.xclbin  # hardware emulation
./host alveo_hls4ml.xclbin  # run on U50
```
## Some detail
```bash
The model contains a zeropad and a conv2d layer.
Both of them are 8 channel input and output.
The input and output port connect to different HBM.
```
