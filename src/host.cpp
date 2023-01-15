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

#include <fstream>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
typedef std::chrono::high_resolution_clock Clock;

#include "xcl2.hpp"
#include <vector>
#include "kernel_params.h"

#define STRINGIFY2(var) #var
#define STRINGIFY(var) STRINGIFY2(var)


static uint64_t get_duration_ns(const std::vector<cl::Event>& events) {
    uint64_t duration = 0;
    for (size_t i=0; i<events.size(); i++) {
        uint64_t start, end;
        events[i].getProfilingInfo<uint64_t>(CL_PROFILING_COMMAND_START, &start);
        events[i].getProfilingInfo<uint64_t>(CL_PROFILING_COMMAND_END, &end);
        duration += end - start;
    }
    return duration;
}

int main(int argc, char** argv)
{

    int nevents = 1;
    cl_int err;
    cl::Kernel krnl_aws_hls4ml;
    std::string datadir = STRINGIFY(HLS4ML_DATA_DIR);
    std::string xclbinFilename = "";
    if (argc > 1) xclbinFilename = argv[1];
    if (argc > 2) nevents = atoi(argv[2]);
    if (argc > 3) datadir = argv[3];
    std::cout << "Will run " << nevents << " time(s), using " << datadir << " to get input features and output predictions (tb_input_features.dat and tb_output_predictions.dat)" << std::endl;

    size_t vector_size_in_bytes = sizeof(input_group) *(IN_STREAM_LEN*DATA_SIZE_IN/8)*DATA_SET;
    size_t vector_size_out_bytes = sizeof(bigdata_t) * (OUT_STREAM_LEN*DATA_SIZE_OUT)*DATA_SET;
    // Allocate Memory in Host Memory
    // When creating a buffer with user pointer (CL_MEM_USE_HOST_PTR), under the hood user ptr 
    // is used if it is properly aligned. when not aligned, runtime had no choice but to create
    // its own host side buffer. So it is recommended to use this allocator if user wish to
    // create buffer using CL_MEM_USE_HOST_PTR to align user buffer to page boundary. It will 
    // ensure that user buffer is used when user create Buffer/Mem object with CL_MEM_USE_HOST_PTR 
    std::vector<input_group,aligned_allocator<input_group>> source_in( (IN_STREAM_LEN*DATA_SIZE_IN/8)*DATA_SET );
    std::vector<bigdata_t,aligned_allocator<bigdata_t>> source_hw_results( (OUT_STREAM_LEN*DATA_SIZE_OUT)*DATA_SET );

    //Reset the input data
    for(int i0 = 0; i0 < (IN_STREAM_LEN*DATA_SIZE_IN/8)*DATA_SET; i0++) {
        for(int i1 = 0; i1 < 8; i1++) {
            source_in[i0].layer[i1] = 0;
        }
    }

    //Reset the output result
    for(int j = 0 ; j < (OUT_STREAM_LEN*DATA_SIZE_OUT)*DATA_SET ; j++){
        source_hw_results[j] = 0;
    }

//=====================================================
//Find device & Load xclbin file & Program device
//=====================================================

    // OPENCL HOST CODE AREA START
    // get_xil_devices() is a utility API which will find the xilinx
    // platforms and will return list of devices connected to Xilinx platform
    std::vector<cl::Device> devices = xcl::get_xil_devices();
    cl::Device device = devices[0];

    cl::Context context(device);
    cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE);
    std::string device_name = device.getInfo<CL_DEVICE_NAME>(); 
    std::cout << "Found Device=" << device_name.c_str() << std::endl;
    
    cl::Program::Binaries bins;
    // Load xclbin
    std::cout << "Loading: '" << xclbinFilename << "'\n";
    std::ifstream bin_file(xclbinFilename, std::ifstream::binary);
    bin_file.seekg (0, bin_file.end);
    unsigned nb = bin_file.tellg();
    bin_file.seekg (0, bin_file.beg);
    char *buf = new char [nb];
    bin_file.read(buf, nb);
    // Create Program from Binary File
    bins.push_back({buf,nb});
    
    // Program the device
    bool valid_device = false;
    cl::Program program(context, {device}, bins, nullptr, &err);
    if (err != CL_SUCCESS) {
        std::cout << "Failed to program device with xclbin file!\n";
    }else {
        std::cout <<"program successful!\n";
        
        std::string cu_id = std::to_string(1);
        std::string krnl_name_full = "alveo_hls4ml";
        printf("Creating a kernel [%s] for CU(%d)\n", krnl_name_full.c_str(), 0);
        krnl_aws_hls4ml = cl::Kernel(program,"alveo_hls4ml");
        valid_device = true;
    }
    if (!valid_device) {
        std::cout << "Failed to program any device found, exit!\n";
        exit(EXIT_FAILURE);
    }

//=====================
//Create buffer
//=====================

    // Allocate Buffer in Global Memory
    // Buffers are allocated using CL_MEM_USE_HOST_PTR for efficient memory and 
    // Device-to-host communication
    cl::Buffer buffer_in   (context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 
            vector_size_in_bytes, source_in.data());
    cl::Buffer buffer_output(context,CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, 
            vector_size_out_bytes, source_hw_results.data());

    int narg = 0;
    krnl_aws_hls4ml.setArg(narg++, buffer_in);
    krnl_aws_hls4ml.setArg(narg++, buffer_output);

    auto t1 = Clock::now();
    auto t2 = Clock::now();
    auto t3 = Clock::now();
    auto t4 = Clock::now();
    auto t5 = Clock::now();


//=====================
//Input
//=====================

    // Load input data from text file
    std::ifstream fin(datadir+"/tb_input_features.dat");
    // Load predictions from text file
    std::ifstream fpr(datadir+"/tb_output_predictions.dat");
    // Open output file
    std::ofstream fout;
    std::ofstream fout3;
    std::ofstream fout4;
    fout.open("tb_output_data.dat");
    fout3.open("tb_output_data3.dat");
    fout4.open("tb_output_data4.dat");
    
    std::string iline;
    std::string pline;
    
    int exp_times = 0;

    // Flag for success/failure of finding data files
    if (!(fin.is_open()) || !(fpr.is_open())) {
        std::cout << "Unable to open input/predictions file, using random input" << std::endl;
        exit(EXIT_FAILURE);
    }
    else std::cout <<"successfully open input and output file"<<std::endl;
    
    // Get inputs/predictions from files
    if(fin.is_open() && fpr.is_open()){
      while(std::getline(fin,iline) && std::getline(fpr,pline)) {
        
        std::cout << "Processing event " << exp_times << std::endl;
        fout << "Processing event " << exp_times << "\n";
        exp_times++;
        
        // Here is input.
        std::vector<float> in;
        char* current;
        char* cstr;
        std::cout <<"in push back"<<std::endl;
        for (int k=0; k< DATA_SET; k++){
            cstr=const_cast<char*>(iline.c_str());
            current=strtok(cstr," ");
            while(current!=NULL){
                in.push_back(atof(current));
                current=strtok(NULL," ");
            }
            std::getline(fin,iline);
        }
        
        //Here is the corresponding output(correct one)
        std::vector<float> pr;
        std::cout <<"pr push back"<<std::endl;
        for (int k=0; k< DATA_SET; k++){
            cstr=const_cast<char*>(pline.c_str());
            current=strtok(cstr," ");
            while(current!=NULL){
                pr.push_back(atof(current));
                current=strtok(NULL," ");
            }
            std::getline(fpr,pline);
        }
        //Send into buffer
        for(int i0 = 0; i0 < (IN_STREAM_LEN*DATA_SIZE_IN/8)*DATA_SET; i0++) {
            for(int i1 = 0; i1 < 8; i1++) {
               source_in[i0].layer[i1] = (bigdata_t)in[i0*8+i1];
            }
        }
        //Reset the output result
        for(int j = 0 ; j < (OUT_STREAM_LEN*DATA_SIZE_OUT)*DATA_SET ; j++){
            source_hw_results[j] = 0;
        }

//========================
//Start to run on FPGA
//========================
        std::vector<cl::Event> events_enqueueTask(1);
        t1 = Clock::now();
        // Copy input data to device global memory
        q.enqueueMigrateMemObjects({buffer_in},0/* 0 means from host*/);
        // Launch the Kernel
        // For HLS kernels global and local size is always (1,1,1). So, it is recommended
        // to always use enqueueTask() for invoking HLS kernel
        q.enqueueTask(krnl_aws_hls4ml,NULL,&events_enqueueTask[0]);
        // Copy Result from Device Global Memory to Host Local Memory
        q.enqueueMigrateMemObjects({buffer_output},CL_MIGRATE_MEM_OBJECT_HOST);
        // Check for any errors from the command queue
        q.finish();
        t5 = Clock::now();
        
        //print timing
        std::cout << "FPGA time: " << std::chrono::duration_cast<std::chrono::nanoseconds>(t5 - t1).count() << " ns" << std::endl;
        fout << "FPGA time: " << std::chrono::duration_cast<std::chrono::nanoseconds>(t5 - t1).count() << " ns \n";
        auto duration_ns = get_duration_ns(events_enqueueTask); 
        std::cout << "kernel(func) time: " << duration_ns << " ns" << std::endl;
        fout << "kernel(func) time: " << duration_ns << " ns \n";
        fout3 << std::chrono::duration_cast<std::chrono::nanoseconds>(t5 - t1).count() << "\n";
        fout4 << duration_ns << "\n";
//=====================
//Output result
//=====================

        std::cout<<"Predictions: \n";
        fout <<"Predictions:  \n";
        for(int i=0;i<(OUT_STREAM_LEN*DATA_SIZE_OUT)*DATA_SET ;i++){
            std::cout << pr[i] << " ";
            fout << pr[i] << " ";
        }
        std::cout << std::endl;
        fout<<"\n";

        std::cout<<"Quantized predictions: \n";
        fout <<"Quantized predictions: \n";
        for(int i=0;i<(OUT_STREAM_LEN*DATA_SIZE_OUT)*DATA_SET ;i++){
            std::cout << source_hw_results[i]<< " ";
            fout << source_hw_results[i] << " "; 
        }
        std::cout << std::endl;
        fout << "\n\n";
        std::cout<<"---- END EVENT "<<" ----"<<std::endl;

      }
    }

// OPENCL HOST CODE AREA END
    fin.close();
    fpr.close();
    fout.close();

    return EXIT_SUCCESS;
}
