//
//    rfnoc-hls-neuralnet: Vivado HLS code for neural-net building blocks
//
//    Copyright (C) 2017 EJ Kreinar
//
//    This program is free software: you can redistribute it and/or modify
//    it under the terms of the GNU General Public License as published by
//    the Free Software Foundation, either version 3 of the License, or
//    (at your option) any later version.
//
//    This program is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.
//
//    You should have received a copy of the GNU General Public License
//    along with this program.  If not, see <http://www.gnu.org/licenses/>.
//
#include <iostream>

#include "kernel.h"
#include "myproject.h"



void kernel(
	const bigdata_t *in,
	model_default_t w2[576],
    bigdata_t *out       // Output Result
) {


    #pragma HLS DATAFLOW

    bigdata_t in_bigbuf[DATA_SIZE_IN*IN_STREAM_LEN];
    bigdata_t out_bigbuf;
    
    hls::stream<input_t> in_buf;
    hls::stream<result_t> out_buf;

    //If input or output variable is array
    //#pragma HLS ARRAY_PARTITION   variable=in_buf  complete dim=0
    //#pragma HLS ARRAY_PARTITION   variable=out_buf complete dim=0
    #pragma HLS STREAM   variable=in_buf  depth=1000
    #pragma HLS STREAM   variable=out_buf depth=1
    
    //Get data from buffer
    for (int i = 0; i < DATA_SIZE_IN*IN_STREAM_LEN; i++) {
        #pragma HLS LOOP UNROLL
        in_bigbuf[i] = in[i];
    }
    
    //=============================================
    //Input
    //=============================================
    
    input_t tmp;
    for(int i0 = 0; i0 < IN_STREAM_LEN; i0++) { 
        for(int i1 = 0; i1 < DATA_SIZE_IN; i1++) { 
            #pragma HLS UNROLL
            tmp = in_bigbuf[i0*8+i1];
            in_buf.write(tmp);
        }
    }

    //=============================================
    //Start computation
    //=============================================

    std::cout<<"inf start"<<std::endl;
    myproject(in_buf,w2,out_buf);
    std::cout<<"inf end"<<std::endl;

    //=============================================
    //Output
    //=============================================

    for(int i1 = 0; i1 < DATA_SIZE_OUT*OUT_STREAM_LEN; i1++) {
        #pragma HLS UNROLL
        result_t tmp_small = out_buf.read();
        out_bigbuf = tmp_small;
        out[i1] = out_bigbuf;
    }

}
