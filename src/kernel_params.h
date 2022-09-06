#include "ap_fixed.h"
#include "defines.h"

//how many consecutive sets of inputs to run over per kernel execution

#define IN_STREAM_LEN  (N_INPUT_1_1*N_INPUT_2_1)
#define OUT_STREAM_LEN  (OUT_HEIGHT_2*OUT_WIDTH_2)

#define DATA_SIZE_IN  N_INPUT_3_1
#define DATA_SIZE_OUT  N_FILT_2

typedef ap_fixed<16,14> bigdata_t;

struct input_group{
    bigdata_t layer[8];
};