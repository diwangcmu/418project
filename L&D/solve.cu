#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include "readfile.h"

__device__ __inline__ void grid_construct(Grid* this_grid, int size, int player){
    this_grid->size = size;
    if (player == 0){
        this_grid->player = 0;
        this_grid->player_reverse = 1;
    } else {
        this_grid->player = 1;
        this_grid->player_reverse = 0;
    }
    for (int i=0; i<9; i++){
        for (int j=0; j<9; j++){
            this_grid->stones[i][j] = NULL;
        }
    }
} 

__device__ __inline__ void grid_printboard(Grid* this_grid){
    for (int i=0; i<20; i++){
        printf("#");
    }
    printf("\n");
    for (int i = 0; i < this_grid->size; i++){
        printf("#");
        for (int j = 0; j < this_grid->size; j++){
            if (this_grid->stones[i][j] != NULL){
                if (this_grid->stones[i][j]->state == 1){
                    printf("x ");
                } else {
                    printf("o ");
                }
            } else {
                printf(". ");
            }
        }
        printf("#\n");
    }
    for (int i=0; i<20; i++){
        printf("#");
    }
    printf("\n");
}

__device__ __inline__ void chain_construct(Chain* this_chain){

    this_chain->chain_size = 0;
}

__device__ __inline__ void chain_addStone(Chain* this_chain, Stone* s){
    s->chain = this_chain;
    this_chain->stones[this_chain->chain_size] = s;
    this_chain->chain_size += 1;
}

__device__ __inline__ void chain_join(Chain* this_chain, Chain* chain){
    for (unsigned int i = 0; i != chain->chain_size; ++i){
        chain_addStone(this_chain, chain->stones[i]);
    }
}

__device__ __inline__ void stone_construct(Stone* this_stone, int row, int col, int state){
    this_stone->state = state;
    this_stone->row = row;
    this_stone->col = col;
    this_stone->chain = NULL;
}

__device__ __inline__ int grid_getLiberties(Grid* this_grid, Chain* cur_chain){
    int cnt = 0;

    for (unsigned int i=0; i<cur_chain->chain_size; i++){
        Stone* s = cur_chain->stones[i];
        int row = s->row;
        int col = s->col;

        Stone* neighbors[4];
        for (int j=0;j<4;j++){
            neighbors[j] = NULL;
        }

        // Don't check outside the board
        if (row > 0) {
            neighbors[0] = this_grid->stones[row - 1][col];
            if (neighbors[0] == NULL) cnt += 1;
        }
        if (row < this_grid->size - 1) {
            neighbors[1] = this_grid->stones[row + 1][col];
            if (neighbors[1] == NULL) cnt += 1;
        }
        if (col > 0) {
            neighbors[2] = this_grid->stones[row][col - 1];
            if (neighbors[2] == NULL) cnt += 1;
        }
        if (col < this_grid->size - 1) {
            neighbors[3] = this_grid->stones[row][col + 1];
            if (neighbors[3] == NULL) cnt += 1;
        }
    }
    return cnt;
}

__device__ __inline__ int grid_checkStone(Grid* this_grid, Stone* s){
    int flag = 0;
    if (grid_getLiberties(this_grid, s->chain) == 0){
        flag = 1;
        for (unsigned int i=0; i<s->chain->chain_size; i++){
            this_grid->stones[s->chain->stones[i]->row][s->chain->stones[i]->col] = NULL;
        }
    }
    return flag;
}

__device__ __inline__ int grid_try_add(Grid* this_grid, int row, int col, int state){
    int s;
    if (state == 1){
        s = 0;
    } else {
        s = 1;
    }

    Stone* newStone= new Stone;
    stone_construct(newStone, row, col, s);
    this_grid->stones[row][col] = newStone;

    Stone* neighbors[4];
    for (int i=0;i<4;i++){
        neighbors[i] = NULL;
    }

    // Don't check outside the board
    if (row > 0) {
        neighbors[0] = this_grid->stones[row - 1][col];
    }
    if (row < this_grid->size - 1) {
        neighbors[1] = this_grid->stones[row + 1][col];
    }
    if (col > 0) {
        neighbors[2] = this_grid->stones[row][col - 1];
    }
    if (col < this_grid->size - 1) {
        neighbors[3] = this_grid->stones[row][col + 1];
    }

    Stone* neighbor;

    int flag = 1;
    for (int i=0; i<4;i++){
        if (neighbors[i] != NULL){
            neighbor = neighbors[i];
            if (neighbor->state != newStone->state){
                if (grid_getLiberties(this_grid, neighbor->chain) == 0){
                    flag = 0;
                    break;
                }
            }
        }
    }

    this_grid->stones[row][col] = NULL;

    return flag;
}

__device__ __inline__ int grid_checklive(Grid* this_grid, int row, int col, int s){
    if (this_grid->stones[row][col] != NULL){
        return 0;
    }
    Stone* neighbors[4];
    for (int j=0;j<4;j++){
        neighbors[j] = NULL;
    }

    int flag = 1;
    // Don't check outside the board
    if (row > 0) {
        neighbors[0] = this_grid->stones[row - 1][col];
        if (neighbors[0] == NULL || neighbors[0]->state != s) flag = 0;
    }
    if (row < this_grid->size - 1) {
        neighbors[1] = this_grid->stones[row + 1][col];
        if (neighbors[1] == NULL || neighbors[1]->state != s) flag = 0;
    }
    if (col > 0) {
        neighbors[2] = this_grid->stones[row][col - 1];
        if (neighbors[2] == NULL || neighbors[2]->state != s) flag = 0;
    }
    if (col < this_grid->size - 1) {
        neighbors[3] = this_grid->stones[row][col + 1];
        if (neighbors[3] == NULL || neighbors[3]->state != s) flag = 0;
    }

    if (flag == 1){
        return grid_try_add(this_grid, row, col, s);
    }

    return 0;
}

__device__ __inline__ int grid_check_win(Grid* this_grid, int black_count, int* black_stones){
    int row, col;

    int black_flag = 0;
    int eye_count = 0;
    //printf("%d\n", black_stones[0]);
    for (int i=0;i<black_count;i++){
        //printf("%d", black_stones[i]);
        row = black_stones[i] / 9; col = black_stones[i] % 9;
        if (this_grid->stones[row][col] != NULL){
            black_flag = 1;
            //printf("eye_count = %d\n", eye_count);
            //Don't check outside the board
            if (row > 0) {
                if (grid_checklive(this_grid, row-1, col, 1) == 1){
                    eye_count += 1;
                }
            }
            //printf("eye_count = %d\n", eye_count);
            if (row < this_grid->size - 1) {
                if (grid_checklive(this_grid, row+1, col, 1) == 1){
                    eye_count += 1;
                }
            }
            //printf("eye_count = %d\n", eye_count);
            if (col > 0) {
                if (grid_checklive(this_grid, row, col-1, 1) == 1){
                    eye_count += 1;
                }
            }
            //printf("eye_count = %d\n", eye_count);
            if (col < this_grid->size - 1) {
                if (grid_checklive(this_grid, row, col+1, 1) == 1){
                    eye_count += 1;
                }
            }
        } else {
            return -1;
        }
    }

    if (black_flag == 1){
        if (eye_count > 4){    
            return 1;
        } else {
            return 0;
        }
    } else {
        return -1;
    }
}

__device__ __inline__ int grid_addStone(Grid* this_grid, int row, int col, int state){
    Stone* newStone = new Stone;
    stone_construct(newStone, row, col, state);
    this_grid->stones[row][col] = newStone;

    Stone* neighbors[4];
    for (int i=0;i<4;i++){
        neighbors[i] = NULL;
    }

    // Don't check outside the board
    if (row > 0) {
        neighbors[0] = this_grid->stones[row - 1][col];
    }
    if (row < this_grid->size - 1) {
        neighbors[1] = this_grid->stones[row + 1][col];
    }
    if (col > 0) {
        neighbors[2] = this_grid->stones[row][col - 1];
    }
    if (col < this_grid->size - 1) {
        neighbors[3] = this_grid->stones[row][col + 1];
    }

    Chain* current_chain = new Chain;
    chain_construct(current_chain);
    chain_addStone(current_chain, newStone);

    Stone* neighbor;

    int flag = 0;
    for (int i=0; i<4;i++){
        if (neighbors[i] != NULL){
            neighbor = neighbors[i];
            if (neighbor->state != newStone->state){
                if (grid_checkStone(this_grid, neighbor) == 1){
                    flag = 1;
                }
            } else {
                if (neighbor->chain != newStone->chain){
                    chain_join(neighbor->chain, newStone->chain);
                }
            }
        }
    }

    if (grid_getLiberties(this_grid, newStone->chain) == 0 && flag == 0){
        this_grid->stones[row][col] = NULL;
        return 0;
    }
    return 1;
}

__device__ __inline__ int grid_search(Grid* this_grid, int range_count, int* range_stones, int black_count, int* black_stones){
    //grid_printboard(this_grid);
    if (grid_check_win(this_grid, black_count, black_stones) == 0){
        //printf("not win\n");
        int global_flag, flag, result;
        if (this_grid->player == 1) global_flag = -1;
        if (this_grid->player == 0) global_flag = 1;

        for (int i=0; i<range_count; i++){
            int cur = range_stones[i];
            if (this_grid->stones[cur / 9][cur % 9] == NULL){
                this_grid->next_grid[i] = new Grid;
                grid_construct(this_grid->next_grid[i], 9, this_grid->player_reverse);
                for (int r=0; r<9; r++){
                    for (int c=0; c<9; c++){
                        if (this_grid->stones[r][c] != NULL){
                            grid_addStone(this_grid->next_grid[i], r, c, this_grid->stones[r][c]->state);
                        }
                    }
                }
                flag = grid_addStone(this_grid->next_grid[i], range_stones[i] / 9, range_stones[i] % 9, this_grid->player);
                if (flag == 1){
                    result = grid_search(this_grid->next_grid[i], range_count, range_stones, black_count, black_stones);
                    if (this_grid->player == 1 && result == 1){
                        global_flag = 1;
                        return 1;
                        //grid_printboard(this_grid->next_grid[i]);
                    }
                    if (this_grid->player == 0 && result == -1){
                        global_flag = -1;
                        return -1;
                    }
                }
            }
        }
        return global_flag;
    } else {
        //printf("win\n");
        return grid_check_win(this_grid, black_count, black_stones);
    }
}

__global__ void
search_kernel(int* cur_stones, int cur_player, int* result, int range_count, int* range_stones, int black_count, int* black_stones) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index < range_count){
        Grid *nextGrid = new Grid;
        grid_construct(nextGrid, 9, 1);

        for (int r=0; r<9; r++){
            for (int c=0; c<9; c++){
                if (cur_stones[r*9+c] != -1){
                    grid_addStone(nextGrid, r, c, cur_stones[r*9+c]);
                }
            }
        }

        int flag = grid_addStone(nextGrid, range_stones[index] / 9, range_stones[index]% 9, cur_player);
        if (flag == 1){
            int cur_result = grid_search(nextGrid, range_count, range_stones, black_count, black_stones);
            if (cur_player == 1 && cur_result == 1){
                result[index] = 1;
            }
            else if (cur_player == 0 && cur_result == -1){
                result[index] = -1;
            }
            else result[index] = 1;
        }
    }
}

int searchCuda(Grid* curGrid, int range_count, stone_pos* range_stones[], int black_count, stone_pos* black_stones[]) {
    const int threadsPerBlock = 2;
    const int blocks = (range_count + threadsPerBlock - 1) / threadsPerBlock;

    int result[range_count];
    int* device_stones;
    int* device_result;
    int* device_range_stones;
    int* device_black_stones;  

    int cur_stones[81];
    for (int i=0; i<9; i++){
        for (int j=0; j<9; j++){
            if (curGrid->stones[i][j] == NULL){
                cur_stones[i*9+j] = -1;
            } else {
                cur_stones[i*9+j] = curGrid->stones[i][j]->state;
            }
        }
    }
    cudaMalloc(&device_stones, 81 * sizeof(int));
    cudaMemcpy(device_stones, cur_stones, 81 * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc(&device_result, range_count * sizeof(int));
    cudaMalloc(&device_range_stones, range_count * sizeof(int));
    cudaMalloc(&device_black_stones, black_count * sizeof(int));

    size_t limit = 0;
    cudaDeviceGetLimit(&limit, cudaLimitStackSize);
    printf("stack size = %u\n", (unsigned)limit);
    limit = 65535;
    cudaDeviceSetLimit(cudaLimitStackSize, limit);
    cudaDeviceGetLimit(&limit, cudaLimitStackSize);
    printf("stack size = %u\n", (unsigned)limit);

    int cur_range_stones[range_count];
    int cur_black_stones[black_count];
    for (int i=0; i<range_count; i++){
        cur_range_stones[i] = range_stones[i]->row * 9 + range_stones[i]->col;
    }

    for (int i=0; i<black_count; i++){
        cur_black_stones[i] = black_stones[i]->row * 9 + black_stones[i]->col;
    }

    cudaMemcpy(device_range_stones, cur_range_stones, range_count * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_black_stones, cur_black_stones, black_count * sizeof(int), cudaMemcpyHostToDevice);

    for(int i = 0; i < range_count; i++) {
        result[i] = 1;
    }
    cudaMemcpy(device_result, result, range_count * sizeof(int), cudaMemcpyHostToDevice);

    search_kernel<<<blocks, threadsPerBlock>>>(device_stones, curGrid->player, device_result,
         range_count, device_range_stones, black_count, device_black_stones);

    cudaThreadSynchronize();
    cudaMemcpy(result, device_result, range_count * sizeof(int), cudaMemcpyDeviceToHost);

    for(int i = 0; i < range_count; i++) {
        if(result[i] == -1) {
            return -1;
        }
    }
    return 1;
}