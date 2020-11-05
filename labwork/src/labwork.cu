#include <stdio.h>
#include <include/labwork.h>
#include <cuda_runtime_api.h>
#include <omp.h>

#define ACTIVE_THREADS 4

int main(int argc, char **argv) {
    printf("USTH ICT Master 2018, Advanced Programming for HPC.\n");
    if (argc < 2) {
        printf("Usage: labwork <lwNum> <inputImage>\n");
        printf("   lwNum        labwork number\n");
        printf("   inputImage   the input file name, in JPEG format\n");
        return 0;
    }

    int lwNum = atoi(argv[1]);
    std::string inputFilename;

    // pre-initialize CUDA to avoid incorrect profiling
    printf("Warming up...\n");
    char *temp;
    cudaMalloc(&temp, 1024);

    Labwork labwork;
    if (lwNum != 2 ) {
        inputFilename = std::string(argv[2]);
        labwork.loadInputImage(inputFilename);
    }

    printf("Starting labwork %d\n", lwNum);
    Timer timer;
    timer.start();
    switch (lwNum) {
        case 1:
            labwork.labwork1_CPU();
            labwork.saveOutputImage("labwork2-cpu-out.jpg");
            printf("labwork 1 CPU ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
            for (int i = 1; i < 64; i++){
                timer.start();
                labwork.labwork1_OpenMP(i);
                printf("%d: %.1fms\n", i, timer.getElapsedTimeInMilliSec());
            }
            labwork.saveOutputImage("labwork2-openmp-out.jpg");
            break;
        case 2:
            labwork.labwork2_GPU();
            break;
        case 3:
            for (int i = 1; i < 64; i++){
                timer.start();
                labwork.labwork3_GPU(i);
                printf("%d: %.1fms\n", i, timer.getElapsedTimeInMilliSec());
            }
            
            labwork.saveOutputImage("labwork3-gpu-out.jpg");
            break;
        case 4:
            // timer.start();
            // labwork.labwork3_GPU(3);
            // printf("%d: %.1fms\n", 3, timer.getElapsedTimeInMilliSec());
            // timer.start();
            // labwork.labwork4_GPU(3);
            // printf("%dx%d: %.1fms\n",3, 3, timer.getElapsedTimeInMilliSec());

            for (int i = 1; i < 16; i++){
                timer.start();
                labwork.labwork4_GPU(i);
                printf("%dx%d: %.1fms\n",i, i, timer.getElapsedTimeInMilliSec());
            }
 
            labwork.saveOutputImage("labwork4-gpu-out.jpg");
            break;
        case 5:
            // labwork.labwork5_CPU();
            // labwork.saveOutputImage("labwork5-cpu-out.jpg");
            labwork.labwork5_GPU(FALSE);
            labwork.saveOutputImage("labwork5-gpu-out.jpg");
            break;
        case 6:
            labwork.labwork6_GPU();
            labwork.saveOutputImage("labwork6-gpu-out.jpg");
            break;
        case 7:
            labwork.labwork7_GPU();
            printf("[ALGO ONLY] labwork %d ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
            labwork.saveOutputImage("labwork7-gpu-out.jpg");
            break;
        case 8:
            labwork.labwork8_GPU();
            printf("[ALGO ONLY] labwork %d ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
            labwork.saveOutputImage("labwork8-gpu-out.jpg");
            break;
        case 9:
            labwork.labwork9_GPU();
            printf("[ALGO ONLY] labwork %d ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
            labwork.saveOutputImage("labwork9-gpu-out.jpg");
            break;
        case 10:
            labwork.labwork10_GPU();
            printf("[ALGO ONLY] labwork %d ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
            labwork.saveOutputImage("labwork10-gpu-out.jpg");
            break;
    }
    printf("labwork %d ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
}

void Labwork::loadInputImage(std::string inputFileName) {
    inputImage = jpegLoader.load(inputFileName);
}

void Labwork::saveOutputImage(std::string outputFileName) {
    jpegLoader.save(outputFileName, outputImage, inputImage->width, inputImage->height, 90);
}

void Labwork::labwork1_CPU() {
    int pixelCount = inputImage->width * inputImage->height;
    outputImage = static_cast<unsigned char *>(malloc(pixelCount * 3));
    for (int j = 0; j < 100; j++) {     // let's do it 100 times, otherwise it's too fast!
        for (int i = 0; i < pixelCount; i++) {
            outputImage[i * 3] = (unsigned char) (((int) inputImage->buffer[i * 3] + (int) inputImage->buffer[i * 3 + 1] +
                                          (int) inputImage->buffer[i * 3 + 2]) / 3);
            outputImage[i * 3 + 1] = outputImage[i * 3];
            outputImage[i * 3 + 2] = outputImage[i * 3];
        }
    }
}

void Labwork::labwork1_OpenMP(int numThreads) {
    int pixelCount = inputImage->width * inputImage->height;
    outputImage = static_cast<unsigned char *>(malloc(pixelCount * 3));
    omp_set_num_threads(numThreads);
    // do something here
    #pragma omp parallel for
    for (int j = 0; j < 100; j++) {     // let's do it 100 times, otherwise it's too fast!
        for (int i = 0; i < pixelCount; i++) {
            outputImage[i * 3] = (unsigned char) (((int) inputImage->buffer[i * 3] + (int) inputImage->buffer[i * 3 + 1] +
                                          (int) inputImage->buffer[i * 3 + 2]) / 3);
            outputImage[i * 3 + 1] = outputImage[i * 3];
            outputImage[i * 3 + 2] = outputImage[i * 3];
        }
    }
}

int getSPcores(cudaDeviceProp devProp) {
    int cores = 0;
    int mp = devProp.multiProcessorCount;
    switch (devProp.major) {
        case 2: // Fermi
            if (devProp.minor == 1) cores = mp * 48;
            else cores = mp * 32;
            break;
        case 3: // Kepler
            cores = mp * 192;
            break;
        case 5: // Maxwell
            cores = mp * 128;
            break;
        case 6: // Pascal
            if (devProp.minor == 1) cores = mp * 128;
            else if (devProp.minor == 0) cores = mp * 64;
            else printf("Unknown device type\n");
            break;
        default:
            printf("Unknown device type\n");
            break;
    }
    return cores;
}

void Labwork::labwork2_GPU() {
    int nDevices = 0;
    // get all devices
    cudaGetDeviceCount(&nDevices);
    printf("Number total of GPU : %d\n\n", nDevices);
    for (int i = 0; i < nDevices; i++){
        // get informations from individual device
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        // something more here
        printf("GPU name is %s\n",prop.name);
        printf("GPU clock is %d\n",prop.clockRate);
        printf("GPU multi processor count is %d\n",prop.multiProcessorCount);
        printf("GPU core count is %d\n",getSPcores(prop));
        printf("GPU Warp size is %d\n", prop.warpSize);
        printf("------------------\n");
        printf("GPU Memory clock rate is %d\n", prop.memoryClockRate);
        printf("GPU Memory bus width: %d\n", prop.memoryBusWidth);
        printf("------------------\n");
        printf("\n");
    }

}

__global__ void grayscale(uchar3 *input, uchar3 *output) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    output[tid].x = (input[tid].x + input[tid].y +
    input[tid].z) / 3;
    output[tid].z = output[tid].y = output[tid].x;
    }

void Labwork::labwork3_GPU(int blockSize) {
    // Calculate number of pixels
    int pixelCount = inputImage->width * inputImage->height;
    outputImage = static_cast<unsigned char *>(malloc(pixelCount * 3));
    uchar3 *hostInput = (uchar3 *) malloc(pixelCount * sizeof(uchar3));
    uchar3 *hostGray = (uchar3 *) malloc(pixelCount * sizeof(uchar3));
    
    int numBlock = pixelCount / blockSize;

    // Copy input image into uchar3
    for (int i = 0; i < pixelCount; i++) {
        hostInput[i].x = (unsigned char) inputImage->buffer[i * 3];
        hostInput[i].y = (unsigned char) inputImage->buffer[i * 3 + 1];
        hostInput[i].z = (unsigned char) inputImage->buffer[i * 3 + 2];
    }
    // Allocate CUDA memory  
    uchar3 *devInput;
    uchar3 *devGray;

    cudaMalloc(&devInput, pixelCount * sizeof(uchar3));
    cudaMalloc(&devGray, pixelCount * sizeof(uchar3));

    // Copy CUDA Memory from CPU to GPU
    cudaMemcpy(devInput, hostInput,
    pixelCount * sizeof(uchar3),
    cudaMemcpyHostToDevice);

    // Processing
    grayscale<<<numBlock, blockSize>>>(devInput, devGray);

    // Copy CUDA Memory from GPU to CPU
    cudaMemcpy(hostGray, devGray,
        pixelCount * sizeof(uchar3),
        cudaMemcpyDeviceToHost);

    // Cleaning
    cudaFree(devInput);

    // Copy uchar3 into output image
    for (int i = 0; i < pixelCount; i++) {
        outputImage[i * 3] = (unsigned char) (hostGray[i].x);
        outputImage[i * 3 + 1] = outputImage[i * 3];
        outputImage[i * 3 + 2] = outputImage[i * 3];
    }
}

__global__ void grayscale2(uchar3 *input, uchar3 *output, int w, int h) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (y >= h ||  x >= w) return;
    output[x + y*w].x = (input[x + y*w].x + input[x + y*w].y +
    input[x + y*w].z) / 3;
    output[x + y*w].z = output[x + y*w].y = output[x + y*w].x;
}

void Labwork::labwork4_GPU(int size) {
    // Calculate number of pixels
    int pixelCount = inputImage->width * inputImage->height;
    outputImage = static_cast<unsigned char *>(malloc(pixelCount * 3));
    uchar3 *hostInput = (uchar3 *) malloc(pixelCount * sizeof(uchar3));
    uchar3 *hostGray = (uchar3 *) malloc(pixelCount * sizeof(uchar3));
    
    dim3 gridSize = dim3(inputImage->width / size + 1, inputImage->height/size + 1);
    dim3 blockSize = dim3(size, size);

    // Copy input image into uchar3
    for (int i = 0; i < pixelCount; i++) {
        hostInput[i].x = (unsigned char) inputImage->buffer[i * 3];
        hostInput[i].y = (unsigned char) inputImage->buffer[i * 3 + 1];
        hostInput[i].z = (unsigned char) inputImage->buffer[i * 3 + 2];
    }
    // Allocate CUDA memory  
    uchar3 *devInput;
    uchar3 *devGray;

    cudaMalloc(&devInput, pixelCount * sizeof(uchar3));
    cudaMalloc(&devGray, pixelCount * sizeof(uchar3));

    // Copy CUDA Memory from CPU to GPU
    cudaMemcpy(devInput, hostInput,
    pixelCount * sizeof(uchar3),
    cudaMemcpyHostToDevice);

    // Processing
    grayscale2<<<gridSize, blockSize>>>(devInput, devGray, inputImage->width,inputImage->height);

    // Copy CUDA Memory from GPU to CPU
    cudaMemcpy(hostGray, devGray,
        pixelCount * sizeof(uchar3),
        cudaMemcpyDeviceToHost);

    // Cleaning
    cudaFree(devInput);

    // Copy uchar3 into output image
    for (int i = 0; i < pixelCount; i++) {
        outputImage[i * 3] = (unsigned char) (hostGray[i].x);
        outputImage[i * 3 + 1] = outputImage[i * 3];
        outputImage[i * 3 + 2] = outputImage[i * 3];
    }
}

void Labwork::labwork5_CPU() {
    float filter []={
        0,0,1,2,1,0,0,0,
        3,13,22,13,3,0,
        1,13,59,97,59,13,1,
        2,22,97,159,97,22,2,
        1,13,59,97,59,13,1,
        3,13,22,13,3,0,
        0,0,1,2,1,0,0,0,
    };
    labwork1_CPU();
    unsigned char * grayImage = outputImage;
    int pixelCount = inputImage->width * inputImage->height;
    outputImage = static_cast<unsigned char *>(malloc(pixelCount * 3));
    memset(outputImage, 0, pixelCount*3);
    int w = inputImage->width;
    int h = inputImage->height;
    for (int row = 3; row < h-3; row++){
        for (int col = 3; col < w-3; col++){
            int sum = 0;
            for (int j = -3; j<=3;j++){
                for (int i = -3; i<=3;i++){
                    sum += grayImage[((row+j)*w + col + i)*3] * filter[(j+3)*7 + i +3];
                }
            }
            sum /= 1003;
            outputImage[(row*w +col)*3] = sum;
            outputImage[(row*w +col)*3 + 1] = sum;
            outputImage[(row*w +col)*3 + 2] = sum;
        }
    }
}

__global__ void blur_gpu(uchar3 *input, uchar3 *output, int w, int h) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    
    if (y >= h ||  x >= w) return;

    float filter []={
        0,0,1,2,1,0,0,0,
        3,13,22,13,3,0,
        1,13,59,97,59,13,1,
        2,22,97,159,97,22,2,
        1,13,59,97,59,13,1,
        3,13,22,13,3,0,
        0,0,1,2,1,0,0,0,
    };

    int sum = 0;
    for (int j = -3; j<=3;j++){
        for (int i = -3; i<=3;i++){
            sum += input[((y+j)*w + x + i)].x * filter[(j+3)*7 + i +3];
        }
    }
    sum /= 1003;

    output[x + y*w].z = output[x + y*w].y = output[x + y*w].x = sum;
}

void Labwork::labwork5_GPU(bool shared) {
    int size = 7;
    labwork4_GPU(size);
    unsigned char * grayImage = outputImage;
    int pixelCount = inputImage->width * inputImage->height;
    outputImage = static_cast<unsigned char *>(malloc(pixelCount * 3));
    memset(outputImage, 0, pixelCount*3);

    uchar3 *hostGray = (uchar3 *) malloc(pixelCount * sizeof(uchar3));
    
    dim3 gridSize = dim3(inputImage->width / size + 1, inputImage->height/size + 1);
    dim3 blockSize = dim3(size, size);

    // Allocate CUDA memory  
    uchar3 *devInput;
    uchar3 *devGray;

    cudaMalloc(&devInput, pixelCount * sizeof(uchar3));
    cudaMalloc(&devGray, pixelCount * sizeof(uchar3));

    // Copy CUDA Memory from CPU to GPU
    cudaMemcpy(devInput, grayImage,
    pixelCount * sizeof(uchar3),
    cudaMemcpyHostToDevice);

    // Processing
    blur_gpu<<<gridSize, blockSize>>>(devInput, devGray, inputImage->width,inputImage->height);

    // Copy CUDA Memory from GPU to CPU
    cudaMemcpy(hostGray, devGray,
        pixelCount * sizeof(uchar3),
        cudaMemcpyDeviceToHost);

    // Cleaning
    cudaFree(devInput);

    // Copy uchar3 into output image
    for (int i = 0; i < pixelCount; i++) {
        outputImage[i * 3] = (unsigned char) (hostGray[i].x);
        outputImage[i * 3 + 1] = outputImage[i * 3];
        outputImage[i * 3 + 2] = outputImage[i * 3];
    }
}

void Labwork::labwork6_GPU() {
}

void Labwork::labwork7_GPU() {
}

void Labwork::labwork8_GPU() {
}

void Labwork::labwork9_GPU() {

}

void Labwork::labwork10_GPU(){
}


























