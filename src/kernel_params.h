#include "ap_fixed.h"
#include "defines.h"

//how many consecutive sets of inputs to run over per kernel execution

#define IN 10
#define OUT 10

typedef ap_fixed<16,8> bigdata_t;
