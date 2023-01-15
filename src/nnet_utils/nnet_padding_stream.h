#ifndef NNET_PADDING_STREAM_H_
#define NNET_PADDING_STREAM_H_

#include <math.h>

namespace nnet {

template<class res_T, typename CONFIG_T>
void fill_zero(hls::stream<res_T> &res) {
    #pragma HLS INLINE
    res_T res_part;
	for (int c = 0; c < CONFIG_T::n_chan; c++) {
        #pragma HLS UNROLL
	    res_part[c] = 0;
    }
    res.write(res_part);
}

template<class res_T, typename CONFIG_T>
void fill_zero_ss(hls::stream<res_T> &res) {
#pragma HLS INLINE
	res_T res_part;
	
	for (int c = 0; c < CONFIG_T::n_chan; c++) {
	#pragma HLS PIPELINE
		res_part = 0;
		res.write(res_part);
	}
}

template<class data_T, class res_T, typename CONFIG_T>
void fill_data(hls::stream<data_T> &data, hls::stream<res_T> &res) {
    #pragma HLS INLINE
    data_T data_part = data.read();
    res_T res_part;
    for (int c = 0; c < CONFIG_T::n_chan; c++) {
        #pragma HLS UNROLL
        res_part[c] = data_part[c];
    }
    res.write(res_part);
}

template<class data_T, class res_T, typename CONFIG_T>
void fill_data_ss(hls::stream<data_T> &data, hls::stream<res_T> &res) {
    	#pragma HLS INLINE

	for (int c = 0; c < CONFIG_T::n_chan; c++) {
	#pragma HLS PIPELINE
		data_T data_part = data.read();
		res_T res_part = data_part;
		res.write(res_part);
    	}

}

template<class data_T, class res_T, typename CONFIG_T>
void zeropad1d_cl(
    hls::stream<data_T> &data,
    hls::stream<res_T>  &res
) {
    PadLeft: for (int i = 0; i < CONFIG_T::pad_left; i++) {
        fill_zero<res_T, CONFIG_T>(res);
    }

    CopyMain: for (int i = 0; i < CONFIG_T::in_width; i++) {
        fill_data<data_T, res_T, CONFIG_T>(data, res);
    }

    PadRight: for (int i = 0; i < CONFIG_T::pad_right; i++) {
        fill_zero<res_T, CONFIG_T>(res);
    }
}

template<class data_T, class res_T, typename CONFIG_T>
void zeropad2d_cl(
    hls::stream<data_T> &data,
    hls::stream<res_T>  &res
) {

    PadTop: for (int i = 0; i < CONFIG_T::pad_top; i++) {
        PadTopWidth: for (int j = 0; j < CONFIG_T::out_width; j++) {
            fill_zero<res_T, CONFIG_T>(res);
        }
    }

    PadMain: for (int i = 0; i < CONFIG_T::in_height; i++) {
        PadLeft: for (int j = 0; j < CONFIG_T::pad_left; j++) {
            fill_zero<res_T, CONFIG_T>(res);
        }
        CopyMain: for (int j = 0; j < CONFIG_T::in_width; j++) {
            fill_data<data_T, res_T, CONFIG_T>(data, res);
        }
        PadRight: for (int j = 0; j < CONFIG_T::pad_right; j++) {
            fill_zero<res_T, CONFIG_T>(res);
        }
    }

    PadBottom: for (int i = 0; i < CONFIG_T::pad_bottom; i++) {
        PadBottomWidth: for (int j = 0; j < CONFIG_T::out_width; j++) {
            fill_zero<res_T, CONFIG_T>(res);
        }
    }
}

template<class data_T, class res_T, typename CONFIG_T>
void zeropad2d_cl_ss(
    hls::stream<data_T> &data,
    hls::stream<res_T>  &res
) {

    PadTop: for (int i = 0; i < CONFIG_T::pad_top; i++) {
        PadTopWidth: for (int j = 0; j < CONFIG_T::out_width; j++) {
            fill_zero_ss<res_T, CONFIG_T>(res);
        }
    }

    PadMain: for (int i = 0; i < CONFIG_T::in_height; i++) {
        PadLeft: for (int j = 0; j < CONFIG_T::pad_left; j++) {
            fill_zero_ss<res_T, CONFIG_T>(res);
        }
        CopyMain: for (int j = 0; j < CONFIG_T::in_width; j++) {
            fill_data_ss<data_T, res_T, CONFIG_T>(data, res);
        }
        PadRight: for (int j = 0; j < CONFIG_T::pad_right; j++) {
            fill_zero_ss<res_T, CONFIG_T>(res);
        }
    }

    PadBottom: for (int i = 0; i < CONFIG_T::pad_bottom; i++) {
        PadBottomWidth: for (int j = 0; j < CONFIG_T::out_width; j++) {
            fill_zero_ss<res_T, CONFIG_T>(res);
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//for switch
template<class res_T, typename CONFIG_T>
void fill_zero_single(hls::stream<res_T> res[1]) {
    #pragma HLS INLINE
    res_T res_part;
    for (int c = 0; c < CONFIG_T::n_chan; c++) {
        #pragma HLS PIPELINE
        res_part = 0;
        res[0].write(res_part);
    }
}

template<class res_T, typename CONFIG_T>
void fill_zero_array(hls::stream<res_T> res[CONFIG_T::n_chan]) {
    #pragma HLS INLINE
    res_T res_part = 0;;
    for (int c = 0; c < CONFIG_T::n_chan; c++) {
        #pragma HLS UNROLL
        res[c].write(res_part);
    }
}

template<class data_T, class res_T, typename CONFIG_T>
void fill_data_single(hls::stream<data_T> data[1], hls::stream<res_T> res[1]) {
    #pragma HLS INLINE
    for (int c = 0; c < CONFIG_T::n_chan; c++) {
        #pragma HLS PIPELINE
        data_T data_part = data[0].read();
        res_T res_part = data_part;
        res[0].write(res_part);
    }

}

template<class data_T, class res_T, typename CONFIG_T>
void fill_data_array(hls::stream<data_T> data[CONFIG_T::n_chan], hls::stream<res_T> res[CONFIG_T::n_chan]) {
    #pragma HLS INLINE
    data_T data_part[CONFIG_T::n_chan];
    #pragma HLS ARRAY_PARTITION variable=data_part complete
    
    for (int c = 0; c < CONFIG_T::n_chan; c++) {
        #pragma HLS UNROLL
        data_part[c] = data[c].read();
    }
    
    for (int c = 0; c < CONFIG_T::n_chan; c++) {
        #pragma HLS UNROLL
        res_T res_part = data_part[c];
        res[c].write(res_part);
    }
}

template<class data_T, class res_T, typename CONFIG_T>
void zeropad2d_cl_single(
    hls::stream<data_T> data[1],
    hls::stream<res_T>  res[1]
) {

    PadTop: for (int i = 0; i < CONFIG_T::pad_top; i++) {
        PadTopWidth: for (int j = 0; j < CONFIG_T::out_width; j++) {
            fill_zero_single<res_T, CONFIG_T>(res);
        }
    }

    PadMain: for (int i = 0; i < CONFIG_T::in_height; i++) {
        PadLeft: for (int j = 0; j < CONFIG_T::pad_left; j++) {
            fill_zero_single<res_T, CONFIG_T>(res);
        }
        CopyMain: for (int j = 0; j < CONFIG_T::in_width; j++) {
            fill_data_single<data_T, res_T, CONFIG_T>(data, res);
        }
        PadRight: for (int j = 0; j < CONFIG_T::pad_right; j++) {
            fill_zero_single<res_T, CONFIG_T>(res);
        }
    }

    PadBottom: for (int i = 0; i < CONFIG_T::pad_bottom; i++) {
        PadBottomWidth: for (int j = 0; j < CONFIG_T::out_width; j++) {
            fill_zero_single<res_T, CONFIG_T>(res);
        }
    }
}

template<class data_T, class res_T, typename CONFIG_T>
void zeropad2d_cl_array(
    hls::stream<data_T> data[CONFIG_T::n_chan],
    hls::stream<res_T>  res[CONFIG_T::n_chan]
) {

    PadTop: for (int i = 0; i < CONFIG_T::pad_top; i++) {
        PadTopWidth: for (int j = 0; j < CONFIG_T::out_width; j++) {
            fill_zero_array<res_T, CONFIG_T>(res);
        }
    }

    PadMain: for (int i = 0; i < CONFIG_T::in_height; i++) {
        PadLeft: for (int j = 0; j < CONFIG_T::pad_left; j++) {
            fill_zero_array<res_T, CONFIG_T>(res);
        }
        CopyMain: for (int j = 0; j < CONFIG_T::in_width; j++) {
            fill_data_array<data_T, res_T, CONFIG_T>(data, res);
        }
        PadRight: for (int j = 0; j < CONFIG_T::pad_right; j++) {
            fill_zero_array<res_T, CONFIG_T>(res);
        }
    }

    PadBottom: for (int i = 0; i < CONFIG_T::pad_bottom; i++) {
        PadBottomWidth: for (int j = 0; j < CONFIG_T::out_width; j++) {
            fill_zero_array<res_T, CONFIG_T>(res);
        }
    }
}


template<class data_T, class res_T, typename CONFIG_T>
void zeropad2d_cl_switch(
    hls::stream<data_T> data[CONFIG_T::data_transfer_out],
    hls::stream<res_T>  res[CONFIG_T::data_transfer_out]
) {
    #pragma HLS inline region
    if(CONFIG_T::data_transfer_out == 1){
        zeropad2d_cl_single<data_T, res_T, CONFIG_T>(data, res);
    }else {
        zeropad2d_cl_array<data_T, res_T, CONFIG_T>(data, res);
    }
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


}

#endif