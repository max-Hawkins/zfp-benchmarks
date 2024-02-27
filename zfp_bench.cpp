#include "cuda.h"
#include <cuda_runtime.h>
#include <iostream>
#include <time.h>
#include <zfp.h>
#include "zfp/array.h"
#include <stdio.h>
#include <unistd.h>
#include <iostream>
#include <cstring>
// g++ -I/nethome/ehong62/zfp/include -L/nethome/ehong62/zfp/build/lib zfp_test.cpp -lzfp

// Timing helper
long long diff(timespec start, timespec end)
{
	timespec temp;
	if ((end.tv_nsec-start.tv_nsec)<0) {
		temp.tv_sec = end.tv_sec-start.tv_sec-1;
		temp.tv_nsec = 1000000000+end.tv_nsec-start.tv_nsec;
	} else {
		temp.tv_sec = end.tv_sec-start.tv_sec;
		temp.tv_nsec = end.tv_nsec-start.tv_nsec;
	}
	return temp.tv_sec * 1000000000 + temp.tv_nsec;
}

int main(int argc, char **argv)
{
    std::cout << "---ZFP Benchmarking---\n";

    // Output file
    FILE *data_file;
    bool data_file_preexists = 0;
    char data_filename[50] = "zfp_bench.csv";

    if (access(data_filename, F_OK) == 0) {
        data_file_preexists = 1;
        data_file = fopen(data_filename, "a");
    } else {
        data_file_preexists = 0;
        data_file = fopen(data_filename, "w");
    }

    if (data_file == NULL) {
        printf("Error opening file!");
        return 1;
    }

    zfp_exec_policy exec_policy = zfp_exec_serial;
    int num_omp_threads = 8;
    int comp_mode = -1;
    float tolerance = 1;
    double rate = 16;
    int precision = 1;
    unsigned int ndims = 2;
    int halo_x = 4;
    int halo_y = 4;

    zfp_type type = zfp_type_double;

    // Parse command line args
   for (int i = 0; i < argc; ++i){
        std::string arg = argv[i];
        // Help Message
        if (arg.rfind("-h", 0) == 0 || arg.rfind("--help", 0) == 0) {
            printf("ZFP Benchmarking\n");
            printf("Available command-line options: \n");
            printf("\t--nx=\n\t\tSet the number of halo region cells in the X-direction\n");
            printf("\t--ny=\n\t\tSet the number of halo region cells in the Y-direction\n");
            printf("\t--mode=\n\t\tSet the compression mode for ZFP (ACC, RATE, PREC, or REV)\n");
            printf("\t--tolerance=\n\t\tSet the fixed-accuracy tolerance\n");
            printf("\t--rate=\n\t\tSet the fixed-rate rate\n");
            printf("\t--precision=\n\t\tSet the fixed-precision precision\n");
            printf("\t--exec=\n\t\tSet the execution mode for ZFP (single-threaded 'CPU', multithreaded 'OMP', or 'CUDA')\n");
            printf("\t--threads=\n\t\tSet the number of OpenMP threads to use.\n");
            // printf("\t--out=\n\t\tSet the output file to save raw data into\n");
            return 0;
        }
        if (arg.rfind("--nx=", 0) == 0) {
            halo_x = std::stoi(arg.substr(5));
        }
        if (arg.rfind("--ny=", 0) == 0) {
            halo_y = std::stoi(arg.substr(5));
        }
        if (arg.rfind("--threads=", 0) == 0) {
            num_omp_threads = std::stoi(arg.substr(10));
        }
        if (arg.rfind("--tolerance=", 0) == 0) {
            tolerance = std::stof(arg.substr(12));
        }
        if (arg.rfind("--rate=", 0) == 0) {
            rate = std::stod(arg.substr(7));
        }
        if (arg.rfind("--precision=", 0) == 0) {
            precision = std::stoi(arg.substr(12));
        }

        if (arg.rfind("--mode=", 0) == 0) {
            if(strcmp(arg.substr(7).c_str(), "ACC")==0){
                comp_mode = zfp_mode_fixed_accuracy;
            } else if(strcmp(arg.substr(7).c_str(), "RATE")==0){
                comp_mode = zfp_mode_fixed_rate;
            }else if(strcmp(arg.substr(7).c_str(), "PREC")==0){
                comp_mode = zfp_mode_fixed_precision;
            }else if(strcmp(arg.substr(7).c_str(), "REV")==0){
                comp_mode = zfp_mode_reversible;
            }
            else{
                std::cout << "Compression mode '" << arg.substr(7) << "' selected.\nMust be 'ACC', 'RATE', 'PREC', or 'REV'" << std::endl;
                return -1;
            }
        }
        if (arg.rfind("--exec=", 0) == 0) {
            if(std::strcmp(arg.substr(7).c_str(), "CPU")==0){
                exec_policy = zfp_exec_serial;
            } else if(strcmp(arg.substr(7).c_str(), "OMP")==0){
                exec_policy = zfp_exec_omp;
            }else if(strcmp(arg.substr(7).c_str(), "CUDA")==0){
                exec_policy = zfp_exec_cuda;
            } else{
                std::cout << "Execution Polity '" << arg.substr(7) << "' selected.\nMust be 'CPU', 'OMP', or 'CUDA'" << std::endl;
                return -1;
            }
        }
        // if (arg.rfind("--out=", 0) == 0) {
        //     out_folder = arg.substr(6);
        // }
    }

    size_t size = sizeof(double) * halo_x * halo_y;
    double *array, *array_after;
    double *d_before, *d_after;

    cudaError_t status = cudaMallocHost((void**)&array, size);
    if (status != cudaSuccess){
        printf("Error allocating pinned host memory\n");
        std::cout << "Error is: " << cudaGetErrorString(status) << std::endl;
    }
    status = cudaMallocHost((void**)&array_after, size);
    if (status != cudaSuccess)
        printf("Error allocating pinned host memory\n");

    // Allocate vectors in device memory
    cudaMalloc(&d_before, size);
    cudaMalloc(&d_after, size);

    timespec start, end;
    int num_trials = 10;
    int num_warmup = 3; // How many compression/decompression cycles to execute before starting timing
    size_t raw_data_size = halo_x * halo_y * sizeof(double);
    long long total_compress_time, total_decompress_time;

    // initialize metadata for a compressed stream
    zfp_stream *zfp = zfp_stream_open(NULL); // compressed stream and parameters
    switch(comp_mode){
        case zfp_mode_fixed_accuracy:
            printf("Using Fixed-Accuracy Mode\nTolerance: %f\n", tolerance);
            zfp_stream_set_accuracy(zfp, tolerance);
            break;
        case zfp_mode_fixed_precision:
            printf("Using Fixed-Precision Mode\nPrecision: %u\n", precision);
            zfp_stream_set_precision(zfp, precision);
            break;
        case zfp_mode_fixed_rate:
            printf("Using Fixed-Rate Mode\nRate: %f\n", rate);
            zfp_stream_set_rate(zfp, rate, type, ndims, 0);
            break;
        case zfp_mode_reversible:
            printf("Using Reversible Mode\n");
            zfp_stream_set_reversible(zfp);
            break;
    }

    size_t compressed_size;
    float compression_ratio;
    double compression_throughput, decompression_throughput;

    total_compress_time = 0;
    total_decompress_time = 0;
    size_t decompressed_size;

    if (zfp_stream_set_execution(zfp, exec_policy)) {
        // Check execution policy was set correctly
        if(zfp->exec.policy != exec_policy){
            printf("Error setting ZFP execution policy.\n");
            return -1;
        }
        printf("ZFP Execution Policy: %d\n", zfp->exec.policy);

        if(exec_policy == zfp_exec_omp){
            zfp_stream_set_omp_threads(zfp, num_omp_threads);
            int actual_num_omp_threads = zfp_stream_omp_threads(zfp);
            if(actual_num_omp_threads != num_omp_threads){
                printf("Error setting number of OpenMP threads.\n");
                return -1;
            }
            printf("Number of OpenMP Threads: %d\n", actual_num_omp_threads);
        }


        for(int y=0; y < halo_y; y++){
            for(int x=0; x < halo_x; x++){
                array[y*halo_x+x] = rand() / (double) RAND_MAX; //0.1;
                array_after[y*halo_x+x] = 0.0;
                // cout << array[y*halo_x+x] << ", ";
            }
            // cout << endl;
        }

        // Copy vectors from host memory to device memory
        cudaMemcpy(d_before, array, raw_data_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_after, array_after, raw_data_size, cudaMemcpyHostToDevice);

        // initialize metadata for the 3D array a[nz][ny][nx]
        zfp_field* field = zfp_field_2d(d_before, type, halo_x, halo_y); // array metadata

        // allocate buffer for compressed data
        size_t bufsize = zfp_stream_maximum_size(zfp, field);     // capacity of compressed buffer (conservative)
        void* buffer;
        cudaMalloc(&buffer, bufsize);

        // associate bit stream with allocated buffer
        bitstream* stream = stream_open(buffer, bufsize);         // bit stream to compress to
        zfp_stream_set_bit_stream(zfp, stream);                   // associate with compressed stream
        // zfp_field_free(field); // TEST
        // zfp_stream_rewind(zfp);                                   // rewind stream to beginning


        total_compress_time = 0;
        total_decompress_time = 0;

        for(int i=0; i<num_trials+num_warmup; i++){
            // compress array
            zfp_stream_rewind(zfp);                                   // rewind stream to beginning
            clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
            compressed_size = zfp_compress(zfp, field);
            cudaDeviceSynchronize();
            clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &end);

            // Ignore printing/data collection during warmup
            if(i>=num_warmup && compressed_size > 0){
                total_compress_time += diff(start,end);
                compression_ratio = raw_data_size / (float)compressed_size;
                compression_throughput = raw_data_size / (float)diff(start,end);

                // printf("\nData Size: %d -> %d\nCompression Ratio: %f\nCompression Time: %lld\nCompression Throughput: %f GB/s\n",
                //         raw_data_size,
                //         compressed_size,
                //         compression_ratio,
                //         diff(start,end),
                //         compression_throughput);
            }

            // Decompress
            zfp_stream_rewind(zfp);

            // zfp_field* field_decompress = zfp_field_2d(d_after, type, halo_x, halo_y); // array metadata
            // bufsize_s = zfp_stream_maximum_size(zfp, field_decompress);     // capacity of compressed buffer (conservative)
            // void* buffer_decompress;
            // cudaMalloc(&buffer_decompress, bufsize_s);
            cudaDeviceSynchronize();

            clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
            decompressed_size = zfp_decompress(zfp, field);
            cudaDeviceSynchronize();
            clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &end);

            // Ignore printing/data collection during warmup
            if(i>=num_warmup && decompressed_size > 0){
                total_decompress_time += diff(start,end);
                decompression_throughput = raw_data_size / (float)diff(start,end);

                // printf("Decompression Time: %d\nDecompression Throughput: %f GB/s\n",
                //         diff(start,end),
                //         decompression_throughput);
            }
        }
    }

    double avg_compression_time = total_compress_time / (double)num_trials;
    double avg_decompression_time = total_decompress_time / (double)num_trials;
    compression_throughput = raw_data_size / avg_compression_time;
    decompression_throughput = raw_data_size / avg_decompression_time;

    if(compressed_size == 0){
        printf("Compression failed. No data collected on decompression.\n");
        avg_compression_time = -1;
        compression_throughput = -1;
    }else{
        printf("\n\nAverage Compression time: %.0f ns\nAverage Compression Throughput: %f GB/s\n",
            avg_compression_time,
            compression_throughput);
    }

    if(decompressed_size == 0){
        printf("\nDecompression failed. No data collected on decompression.\n");
        avg_decompression_time = -1;
        decompression_throughput = -1;
    }else{
        printf("\nAverage Decompression time: %.0f ns\nAverage Decompression Throughput: %f GB/s\n",
            avg_decompression_time,
            decompression_throughput);
    }


    // Check decompressed output
    // cout << "After decompression: " << endl;
    // for(int y=0; y < halo_y; y++){
    //     for(int x=0; x < halo_x; x++){
    //         cout << array_after[y*halo_x+x] << ", ";
    //     }
    //     cout << endl;
    // }

    // Output Header Line
    if(!data_file_preexists){
        char header[300] = "Host,Mode,Exec,OpenMPThreads,Tolerance,Precision,Rate,NX,NY,AvgCompTime,AvgCompThroughput,AvgDecompTime,AvgDecompThroughput";
        fprintf(data_file, "%s\n", header);
    }

    char hostname[1024];
    hostname[1023] = '\0';
    gethostname(hostname, 1023);
    fprintf(data_file,
            "%s,%d,%d,%u,%f,%u,%f,%d,%d,%f,%f,%f,%f\n",
            hostname,
            comp_mode,
            exec_policy,
            num_omp_threads,
            tolerance,
            precision,
            rate,
            halo_x,
            halo_y,
            avg_compression_time,
            compression_throughput,
            avg_decompression_time,
            decompression_throughput);
    fclose(data_file);

    printf("CSV data has been successfully written to the file.\n\n\n");



    return 0;
    }