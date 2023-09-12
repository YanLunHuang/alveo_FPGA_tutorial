#ifndef DEFINES_H_
#define DEFINES_H_

#include "ap_int.h"
#include "ap_fixed.h"
#include "nnet_utils/nnet_types.h"
#include <cstddef>
#include <cstdio>

//hls-fpga-machine-learning insert numbers
#define N_INPUT_1_1 3
#define N_INPUT_2_1 3
#define N_INPUT_3_1 8
#define OUT_HEIGHT_4 5
#define OUT_WIDTH_4 5
#define N_CHAN_4 8
#define OUT_HEIGHT_2 3
#define OUT_WIDTH_2 3
#define N_FILT_2 8

//hls-fpga-machine-learning insert layer-precision
typedef ap_fixed<16,14> model_default_t;
typedef ap_fixed<16,14> input_t;
typedef ap_fixed<16,14> layer4_t;
typedef ap_fixed<16,14> result_t;

#endif

/*
//hls-fpga-machine-learning insert layer-precision
typedef ap_fixed<16,14> model_default_t;
typedef nnet::array<ap_fixed<16,14>, 256*1> input_t;
typedef nnet::array<ap_fixed<16,14>, 256*1> layer4_t;
typedef nnet::array<ap_fixed<16,14>, 256*1> layer2_t;

#endif
*/