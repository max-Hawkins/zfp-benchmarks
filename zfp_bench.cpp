#include <iostream>
#include <fstream>
#include <time.h>
#include <zfp.h>
#include "zfp/array.h"

// g++ -I/nethome/ehong62/zfp/include -L/nethome/ehong62/zfp/build/lib zfp_test.cpp -lzfp

using namespace std;

// Timing helper
timespec diff(timespec start, timespec end)
{
    timespec temp;
    if ((end.tv_nsec - start.tv_nsec) < 0)
    {
        temp.tv_sec = end.tv_sec - start.tv_sec - 1;
        temp.tv_nsec = 1000000000 + end.tv_nsec - start.tv_nsec;
    }
    else
    {
        temp.tv_sec = end.tv_sec - start.tv_sec;
        temp.tv_nsec = end.tv_nsec - start.tv_nsec;
    }
    return temp;
}

int main(int argc, char **argv)
{
    ofstream myfile("output.txt");
    if (myfile.is_open())
    {
        time_t rawtime;
        struct tm *timeinfo;

        time(&rawtime);
        timeinfo = localtime(&rawtime);
        myfile << ("Current local time and date: %s", asctime(timeinfo));
        myfile << "ZFP Test:\n";

        float tolerance = 1;
        int nx = 5;
        int ny = 5;
        int halo_width = 2;
        int halo_x = 4; // nx + 2 * halo_width;
        int halo_y = 4; // ny + 2 * halo_width;

        float *array = new float[halo_x * halo_y];
        float *array_after = new float[halo_x * halo_y];

        timespec start, end;
        int num_trials = 100;
        long total_compress_time, total_decompress_time;

        myfile << "Init: " << endl;
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
        zfp_type type = zfp_type_float;                               // array scalar type
        zfp_field *field = zfp_field_2d(array, type, halo_x, halo_y); // array metadata

        // initialize metadata for a compressed stream
        zfp_stream *zfp = zfp_stream_open(NULL); // compressed stream and parameters
        zfp_stream_set_accuracy(zfp, tolerance); // set tolerance for fixed-accuracy mode
        //  zfp_stream_set_precision(zfp, precision);             // alternative: fixed-precision mode
        //  zfp_stream_set_rate(zfp, rate, type, 3, 0);           // alternative: fixed-rate mode

        // allocate buffer for compressed data
        size_t bufsize = zfp_stream_maximum_size(zfp, field); // capacity of compressed buffer (conservative)
        void *buffer = malloc(bufsize);                       // storage for compressed stream

        // associate bit stream with allocated buffer
        bitstream *stream = stream_open(buffer, bufsize); // bit stream to compress to
        zfp_stream_set_bit_stream(zfp, stream);           // associate with compressed stream
        zfp_stream_rewind(zfp);                           // rewind stream to beginning

        size_t zfpsize;
        float compression_ratio;
        total_compress_time = 0;
        size_t bufsize_s, decompressed_size;

        for (int i = 0; i < num_trials; i++)
        {
            // compress array
            clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
            zfpsize = zfp_compress(zfp, field);
            clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &end);
            myfile << "Compress time: " << diff(start, end).tv_sec << " s  :  " << diff(start, end).tv_nsec << " ns \n"
                   << endl;
            total_compress_time += diff(start, end).tv_nsec;

            compression_ratio = (((halo_x * halo_y * 1.0) - zfpsize) / (halo_x * halo_y * 1.0)) * 100.0;
            myfile << "Uncompressed Size: " << halo_x * halo_y << "  Compressed data size: " << zfpsize << "  Compression Ratio: " << compression_ratio << endl;

            // Decompress
            zfp_stream_rewind(zfp);

            zfp_field *field_decompress = zfp_field_2d(array_after, type, halo_x, halo_y); // array metadata
            bufsize_s = zfp_stream_maximum_size(zfp, field_decompress);                    // capacity of compressed buffer (conservative)

            clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
            decompressed_size = zfp_decompress(zfp, field_decompress);
            clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &end);

            myfile << "Decompressed size: " << decompressed_size << endl;
            myfile << "Decompress time: " << diff(start, end).tv_sec << " s   : " << diff(start, end).tv_nsec << " ns" << endl;
            total_decompress_time += diff(start, end).tv_nsec;
            myfile << "Total Decompress time: " << total_decompress_time << endl;
        }
        myfile << "\nAverage Compression time:   " << total_compress_time / num_trials << " ns" << endl;
        myfile << "Average Decompression time: " << total_decompress_time / num_trials << " ns" << endl;
        myfile.close();
    }
    else
    {
        myfile << "Unable to open file";
    }

    // Check decompressed output
    // cout << "After decompression: " << endl;
    // for(int y=0; y < halo_y; y++){
    //     for(int x=0; x < halo_x; x++){
    //         cout << array_after[y*halo_x+x] << ", ";
    //     }
    //     cout << endl;
    // }

    return 0;
}