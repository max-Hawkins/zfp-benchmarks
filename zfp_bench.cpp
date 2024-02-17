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
    size_t size = sizeof(double) * halo_x * halo_y;

    zfp_type type = zfp_type_float;

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


    float *array = new float[halo_x * halo_y];
    float *array_after = new float[halo_x * halo_y];

    timespec start, end;
    int num_trials = 10;
    int num_warmup = 3; // How many compression/decompression cycles to execute before starting timing
    size_t raw_data_size = halo_x * halo_y * sizeof(double);
    long long total_compress_time, total_decompress_time;

    for (int y = 0; y < halo_y; y++)
    {
        for (int x = 0; x < halo_x; x++)
        {
            array[y * halo_x + x] = y * halo_x + x;
            array_after[y * halo_x + x] = 0.0;
            // myfile << array[y*halo_x+x] << ", ";
        }
        // myfile << endl;
    }

    // initialize metadata for the 3D array a[nz][ny][nx]
    zfp_field *field = zfp_field_2d(array, type, halo_x, halo_y); // array metadata

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

    // allocate buffer for compressed data
    size_t bufsize = zfp_stream_maximum_size(zfp, field); // capacity of compressed buffer (conservative)
    void *buffer = malloc(bufsize);                       // storage for compressed stream

    // associate bit stream with allocated buffer
    bitstream *stream = stream_open(buffer, bufsize); // bit stream to compress to
    zfp_stream_set_bit_stream(zfp, stream);           // associate with compressed stream
    zfp_stream_rewind(zfp);                           // rewind stream to beginning

    size_t compressed_size;
    float compression_ratio;
    float compression_throughput, decompression_throughput;

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

        for (int i = 0; i < num_trials; i++)
        {
            // compress array
            clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
            compressed_size = zfp_compress(zfp, field);
            clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &end);

            // Ignore printing/data collection during warmup
            if(i>=num_warmup){
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

            zfp_field *field_decompress = zfp_field_2d(array_after, type, halo_x, halo_y); // array metadata
            bufsize = zfp_stream_maximum_size(zfp, field_decompress);                    // capacity of compressed buffer (conservative)

            clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
            decompressed_size = zfp_decompress(zfp, field_decompress);
            clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &end);

            // Ignore printing/data collection during warmup
            if(i>=num_warmup && decompressed_size > 0){
                total_decompress_time += diff(start,end);
                decompression_throughput = raw_data_size / (double)diff(start,end);

                printf("Decompression Time: %d\nDecompression Throughput: %f GB/s\n",
                        diff(start,end),
                        decompression_throughput);
            }
        }
    }

    double avg_compression_time = total_compress_time / (double)num_trials;
    double avg_decompression_time = total_decompress_time / (double)num_trials;
    compression_throughput = raw_data_size / avg_compression_time;
    decompression_throughput = raw_data_size / avg_decompression_time;

    printf("\n\nAverage Compression time: %.0f ns\nAverage Compression Throughput: %f GB/s\n",
            avg_compression_time,
            compression_throughput);

    if(decompressed_size == 0){
        printf("\nDecompression failed. No data collected on decompression.\n");
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
        char header[300] = "Host,Mode,Exec,Tolerance,Precision,Rate,AvgCompTime,AvgCompThroughput,AvgDecompTime,AvgDecompThroughput";
        fprintf(data_file, "%s\n", header);
    }

    char hostname[1024];
    hostname[1023] = '\0';
    gethostname(hostname, 1023);
    fprintf(data_file,
            "%s,%d,%d,%u,%f,%u,%f,%f,%f,%f,%f\n",
            hostname,
            comp_mode,
            exec_policy,
            num_omp_threads,
            tolerance,
            precision,
            rate,
            avg_compression_time,
            compression_throughput,
            avg_decompression_time,
            decompression_throughput);
    fclose(data_file);

    printf("CSV data has been successfully written to the file.\n\n\n");



    return 0;
}