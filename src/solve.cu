#include "Stone.cpp"
#include "Chain.cpp"
#include "Grid.cpp"
#include <iostream>
#include "readfile.h"

__global__ void
search_kernel(Grid* curGrid, int* result, int range_count, stone_pos* range_stones[], int black_count, stone_pos* black_stones[]) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index < range_count) { //yong yuan dou shi bai
        Grid *nextGrid = new Grid;
        nextGrid->construct(9, 1); //xia yi bu gai hei xia
        for (int r=0; r<9; r++){
            for (int c=0; c<9; c++){
                if (curGrid->stones[r][c] != NULL){
                    nextGrid->addStone(r, c, curGrid->stones[r][c]->state);
                }
            }
        }
        flag = nextGrid->addStone(range_stones[index]->row, range_stones[index]->col, curGrid->player);
        if (flag == 1){
            result = nextGrid->search(range_count, range_stones, black_count, black_stones);
            if (this->player == 1 && result == 1){
                result[index] = 1;
                //nextGrid->printboard();
            }
            else if (this->player == 0 && result == -1){
                result[index] = -1;
                //nextGrid->printboard();
            }
            else result[index] = 1;
        }
    }
}

void searchCuda(Grid* curGrid, int range_count, stone_pos** range_stones, int black_count, stone_pos** black_stones, int* feedback) {
    const int threadsPerBlock = 16;
    const int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    int result[range_count];
    int* device_result;
    int** device_range_stones;
    int** device_black_stones;
    Grid* device_curGrid;
    cudaMalloc(&device_curGrid, 1);
    cudaMalloc(&device_result, range_count);
    cudaMalloc(&device_range_stones, range_count);
    cudaMalloc(&device_black_stones, black_count);

    cudaMemcpy(device_curGrid, curGrid, 1, cudaMemcpyHostToDevice);
    cudaMemcpy(device_range_stones, range_stones, range_count, cudaMemcpyHostToDevice);
    cudaMemcpy(device_black_stones, black_stones, black_count, cudaMemcpyHostToDevice);
    
    search_kernel<<<blocks, threadsPerBlock>>>(device_curGrid, device_result,
         range_count, device_range_stones, black_count, device_black_stones);

    cudaThreadSynchonize();
    cudaMemcpy(result, device_result, range_count, cudaMemcpyDeviceToHost);
    for(int i = 0; i < range_count, i++) {
        if(result[i] == -1) {
            *feedback = -1;
            return;
        }
    }
    *feedback = 1;
}