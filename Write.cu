
#include <cuda_runtime.h>
#include <cufft.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <complex>
#include <chrono> 
#include <fcntl.h>     // shm_open
#include <sys/mman.h>  // mmap, PROT_READ, PROT_WRITE
#include <unistd.h>    // ftruncate, close
#include <cstring>  


/////fft point size per chunk/////
#define CHUNK_SIZE 1024      ///4mb data
/////No of chunks////////
#define No_chunks 4000



//// Kernel for correlations/////////

__global__ void correlation(cufftComplex* fft_data1, const cufftComplex* fft_data2, cufftComplex* crosscorr, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {

        // Perform cross-correlation calculation using shared memory
        cufftComplex value1 = fft_data1[idx];
        cufftComplex value2 = fft_data2[idx];

        // Update cross-correlation results using atomicAdd for thread safety
        atomicAdd(&crosscorr[idx].x, value1.x * value2.x + value1.y * value2.y);  // Real parts
        atomicAdd(&crosscorr[idx].y, value1.y * value2.x - value1.x * value2.y);  // Imaginary parts
    }
}

//  convert int8 data to float
__global__ void convertInt8ToFloat(const int8_t* int_data, float* float_data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        float_data[idx] = static_cast<float>(int_data[idx]);
    }
}

/////stucture for diff datatype varible storege in mem/////////
struct SharedMemory1 {
    int chunk_no;
    char timestamp[100];
    cufftComplex h_data1[CHUNK_SIZE/2];
    cufftComplex h_data2[CHUNK_SIZE/2];
    cufftComplex h_data3[CHUNK_SIZE/2];

};



int main() {



 auto start = std::chrono::high_resolution_clock::now();
    //////////shared mem name/////////
    const char* SHM_NAME_1 = "/shm_autocorr1";
    /////////fun to creat object and read write access//////////

    int shm_fd1 = shm_open(SHM_NAME_1, O_CREAT | O_RDWR, 0666);
    if (shm_fd1 == -1 ) {
        std::cerr << "Error creating shared memory!" << std::endl;
        return -1;
    }

    ///////function for mapping////////////
    SharedMemory1* shm_ptr1 = (SharedMemory1*)mmap(0, sizeof(SharedMemory1), PROT_WRITE, MAP_SHARED, shm_fd1, 0);
    if (shm_ptr1 == MAP_FAILED ) {
        std::cerr << "Error mapping shared memory!" << std::endl;
        return -1;
    }                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           

    // Resize shared memory
    ftruncate(shm_fd1, sizeof(SharedMemory1));


    /////////requied pointers /////////////////
    int8_t *d_data1, *d_data2;
    float *d_data1_float, *d_data2_float;
    cufftComplex *d_fft_data1, *d_fft_data2;
    cufftComplex *d_autocorr1, *d_autocorr2;
    cufftComplex *d_crosscorr;

    cufftComplex* h_data1 = new cufftComplex[CHUNK_SIZE/2];
    cufftComplex* h_data2 = new cufftComplex[CHUNK_SIZE/2];
    cufftComplex* h_data3 = new cufftComplex[CHUNK_SIZE/2];

    ////mem allocation //////////
    cudaMalloc(&d_data1, CHUNK_SIZE * sizeof(int8_t));
    cudaMalloc(&d_data2, CHUNK_SIZE * sizeof(int8_t));
    cudaMalloc(&d_data1_float, CHUNK_SIZE * sizeof(float));
    cudaMalloc(&d_data2_float, CHUNK_SIZE * sizeof(float));
    cudaMalloc(&d_fft_data1, (CHUNK_SIZE/2) * sizeof(cufftComplex));
    cudaMalloc(&d_fft_data2, (CHUNK_SIZE/2) * sizeof(cufftComplex));
    cudaMalloc(&d_autocorr1, (CHUNK_SIZE/2) * sizeof(cufftComplex));
    cudaMalloc(&d_autocorr2, (CHUNK_SIZE/2) * sizeof(cufftComplex));
    cudaMalloc(&d_crosscorr, (CHUNK_SIZE/2) * sizeof(cufftComplex));
  
  
    ///////plan for fft/////////
    cufftHandle plan2;
    // cufftPlan1d(&plan2, BLOCK_SIZE, CUFFT_R2C, N);
    cufftPlan1d(&plan2, CHUNK_SIZE , CUFFT_R2C, 1 );


    ///////vector to read data in readable format//////////
    std::vector<int8_t> data1(CHUNK_SIZE);
    std::vector<int8_t> data2(CHUNK_SIZE);

    ////// Load data from files///////////
    std::ifstream file1("pol1_2024_12_26_11_52_AM.raw", std::ios::binary);
    std::ifstream file2("pol2_2024_12_26_12_26_PM.raw", std::ios::binary);

    ////////no of blocks for kernal excution////////
    int num_blocks = ((CHUNK_SIZE/2) + 255) / 256;

    /////  To Track number of processed batches  ////////////
     int batch_counter = 0;  

while (file1 && file2 ) {  


    ///////////set correlation data to 0 to procced new chunk data ///////
    cudaMemset(d_autocorr1, 0, (CHUNK_SIZE / 2) * sizeof(cufftComplex));
    cudaMemset(d_autocorr2, 0, (CHUNK_SIZE / 2) * sizeof(cufftComplex));
    
    
    
    for (int chunk = 0; chunk < No_chunks; chunk++) {  // Process 4000 chunks in one batch
        if (!(file1.read(reinterpret_cast<char*>(data1.data()), CHUNK_SIZE) &&
              file2.read(reinterpret_cast<char*>(data2.data()), CHUNK_SIZE))) {
            break;
        }

        std::streamsize bytesRead1 = file1.gcount();
        std::streamsize bytesRead2 = file2.gcount();

        if (bytesRead1 < CHUNK_SIZE || bytesRead2 < CHUNK_SIZE) break;

        cudaMemcpy(d_data1, data1.data(), bytesRead1 * sizeof(int8_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_data2, data2.data(), bytesRead2 * sizeof(int8_t), cudaMemcpyHostToDevice);

        convertInt8ToFloat<<<num_blocks, 256>>>(d_data1, d_data1_float, CHUNK_SIZE);
        convertInt8ToFloat<<<num_blocks, 256>>>(d_data2, d_data2_float, CHUNK_SIZE);

        cufftExecR2C(plan2, d_data1_float, d_fft_data1);
        cufftExecR2C(plan2, d_data2_float, d_fft_data2);

        correlation<<<num_blocks, 256>>>(d_fft_data1,d_fft_data1, d_autocorr1, CHUNK_SIZE / 2);
        correlation<<<num_blocks, 256>>>(d_fft_data2,d_fft_data2, d_autocorr2, CHUNK_SIZE / 2);
        correlation<<<num_blocks, 256>>>(d_fft_data1,d_fft_data2, d_crosscorr, CHUNK_SIZE / 2);

        cudaDeviceSynchronize();
        // processedChunks++; 
    }

    //////// Copy processed data back to host
    cudaMemcpy(h_data1, d_autocorr1, (CHUNK_SIZE / 2) * sizeof(cufftComplex), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_data2, d_autocorr2, (CHUNK_SIZE / 2) * sizeof(cufftComplex), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_data3, d_crosscorr, (CHUNK_SIZE / 2) * sizeof(cufftComplex), cudaMemcpyDeviceToHost);

    ///////// Store results in shared memory
    shm_ptr1->chunk_no = batch_counter + 1;
    memcpy(shm_ptr1->h_data1, h_data1, (CHUNK_SIZE / 2) * sizeof(cufftComplex));
    memcpy(shm_ptr1->h_data2, h_data2, (CHUNK_SIZE / 2) * sizeof(cufftComplex));
    memcpy(shm_ptr1->h_data3, h_data3, (CHUNK_SIZE / 2) * sizeof(cufftComplex));

    batch_counter++;  // Move to the next batch
}

    file1.close();
    file2.close();
       

    munmap(shm_ptr1, sizeof(SharedMemory1));
    shm_unlink("/shm_autocorr1");
    shm_ptr1 = nullptr;

    
    // Close shared memory and clear varibles
    close(shm_fd1);
    cudaFree(d_data1);
    cudaFree(d_data2);
    cudaFree(d_data1_float);
    cudaFree(d_data2_float);
    cudaFree(d_fft_data1);
    cudaFree(d_fft_data2);
    cudaFree(d_autocorr1);
    cudaFree(d_autocorr2);
    cudaFree(d_crosscorr);
    delete[] h_data1;
    delete[] h_data2;
    delete[] h_data3;
    cufftDestroy(plan2);
    

    auto stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = stop - start;

    std::cout << "Runtime: " << elapsed.count()<< " sec" << std::endl;
   
    std::cout << "successful 😊 !"<< std::endl;
    
    
    return 0;
} 






