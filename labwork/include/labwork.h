#pragma once

#include <include/jpegloader.h>
#include <include/timer.h>

class Labwork {

private:
    JpegLoader jpegLoader;
    JpegInfo *inputImage;
    unsigned char *outputImage;

public:
    void loadInputImage(std::string inputFileName);
    void saveOutputImage(std::string outputFileName);

    void labwork1_CPU();
    void labwork1_OpenMP(int numThreads);

    void labwork2_GPU();

    void labwork3_GPU(int blockSize);

    void labwork4_GPU(int size);

    void labwork5_CPU();

    void labwork5_GPU(bool shared);

    void labwork6_GPU();

    void labwork7_GPU();

    void labwork8_GPU();

    void labwork9_GPU();

    void labwork10_GPU();
};
