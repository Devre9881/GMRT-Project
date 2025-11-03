#include <iostream>
#include <vector>
#include <chrono>
#include <cufft.h>
#include <sys/mman.h>  
#include <fcntl.h>     
#include <unistd.h>
#include <cstring>
#include <cstdio>  // For popen()

struct SharedMemory {
    int chunk_no;
    char timestamp[100];

    cufftComplex h_data1[512];  
    cufftComplex h_data2[512];
    cufftComplex h_data3[512];  
};

int main() {
    std::string shmName = "/shm_autocorr1";  // Shared memory name

    // Open shared memory
    int fd = shm_open(shmName.c_str(), O_RDONLY, 0666);
    if (fd == -1) {
        std::cerr << "Error: Could not open shared memory." << std::endl;
        return 1;
    }

    // Map shared memory
    SharedMemory* sharedMem = (SharedMemory*)mmap(0, sizeof(SharedMemory), PROT_READ, MAP_SHARED, fd, 0);
    if (sharedMem == MAP_FAILED) {
        std::cerr << "Error: Could not map shared memory." << std::endl;
        close(fd);
        return 1;
    }

    // Open pipe to gnuplot
    FILE* gnuplot = popen("gnuplot -persist", "w");
    if (!gnuplot) {
        std::cerr << "Error: Could not open gnuplot." << std::endl;
        munmap(sharedMem, sizeof(SharedMemory));
        close(fd);
        return 1;
    }

    // Gnuplot settings
    fprintf(gnuplot, "set title 'Real-time Shared Memory Data'\n");
    fprintf(gnuplot, "set xlabel 'Index'\n");
    fprintf(gnuplot, "set ylabel 'Amplitude'\n");
    fprintf(gnuplot, "set grid\n");
    fprintf(gnuplot, "set key outside\n");
    fprintf(gnuplot, "set term x11\n");  // Ensures interactive window
    fflush(gnuplot);

    int last_chunk_no = -1;  // Store last processed chunk number

    while (true) {
        // Check if new data has arrived
        if (sharedMem->chunk_no != last_chunk_no) {
            last_chunk_no = sharedMem->chunk_no;  // Update last processed chunk

            // Start inline data block
            fprintf(gnuplot, "plot '-' using 1:2 with lines title 'h_data1', '-' using 1:2 with lines title 'h_data2'\n");

            // Stream h_data1
            for (int i = 0; i < 500; i++) {
                fprintf(gnuplot, "%d %f\n", i, sharedMem->h_data1[i].x);
            }
            fprintf(gnuplot, "e\n");  // End of first dataset

            // Stream h_data2
            for (int i = 0; i < 512; i++) {
                fprintf(gnuplot, "%d %f\n", i, sharedMem->h_data2[i].x);
            }
            fprintf(gnuplot, "e\n");  // End of second dataset

            fflush(gnuplot);

            std::cout << "Buffer " << sharedMem->chunk_no 
                      << " processed at " << sharedMem->timestamp 
                      << " | 1st_element: " << sharedMem->h_data1[0].x 
                      << std::endl;
        }

//        usleep(50000);  // Sleep for 50ms to reduce CPU usage
    }

    // Cleanup
    pclose(gnuplot);
    munmap(sharedMem, sizeof(SharedMemory));
    close(fd);

    return 0;
}

