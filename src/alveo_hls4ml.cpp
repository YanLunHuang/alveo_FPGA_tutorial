/**********
Copyright (c) 2018, Xilinx, Inc.
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
may be used to endorse or promote products derived from this software
without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
**********/

/*******************************************************************************
Description:
    HLS pragmas can be used to optimize the design : improve throughput, reduce latency and 
    device resource utilization of the resulting RTL code
    This is a wrapper to be used with an hls4ml project to enable proper handling by SDAccel
*******************************************************************************/
#include <iostream>
#include "kernel_params.h"

extern "C" {

void alveo_hls4ml(
    const bigdata_t *in, // Read-Only Vector
    bigdata_t *out       // Output Result
    )
{
    #pragma HLS INTERFACE m_axi port=in  offset=slave bundle=gmem0
    #pragma HLS INTERFACE m_axi port=out offset=slave bundle=gmem1
    #pragma HLS INTERFACE s_axilite port=in   bundle=control
    #pragma HLS INTERFACE s_axilite port=out  bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control

    #pragma HLS DATAFLOW

    bigdata_t in_bigbuf[IN];
    bigdata_t out_bigbuf[OUT];
    
    //Get data from DRAM
    for (int i = 0; i < IN; i++) {
        #pragma HLS PIPELINE II=1
        in_bigbuf[i] = in[i];
    }
    //=============================================
    //Start computation
    //=============================================
    for (int i = 0; i < IN; i++) {
        #pragma HLS PIPELINE II=1
        out_bigbuf[i] = in_bigbuf[i]+i;
    }

    //=============================================
    //Output
    //=============================================
    for (int i = 0; i < OUT; i++) {
        #pragma HLS PIPELINE II=1
        out[i] = out_bigbuf[i];
    }
}
}
