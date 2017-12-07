#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include "readfile.h"

struct GameBoard;
__device__ __inline__ void device_clear_visited(GameBoard* this_board);
__device__ __inline__ void device_board_construct(GameBoard* this_board, int s);
__device__ __inline__ int device_board_addStone(GameBoard* this_board, int row, int col, int state);
__device__ __inline__ void device_delete_stone(GameBoard* this_board, int row, int col);
__device__ __inline__ int device_get_liberties(GameBoard* this_board, int row, int col);
__device__ __inline__ int device_checkStone(GameBoard* this_board, int row, int col, int state);
__device__ __inline__ void device_board_get_terr(GameBoard* this_board);

__device__ __inline__ int d_min_of(int x, int y);
__device__ __inline__ int d_max_of(int x, int y);
__device__ __inline__ int d_abso(int x);
__device__ __inline__ int d_mapping(int dist);

__device__ __inline__
int d_min_of(int x, int y) {
  if(x < y) return x;
  return y;
}

__device__ __inline__
int d_max_of(int x, int y) {
  if(x > y) return x;
  return y;
}

__device__ __inline__
int d_abso(int x) {
  if(x < 0) return -x;
  return x;
}

__device__ __inline__
int d_mapping(int dist) {
  if(dist == 4) return 1;
  if(dist == 3) return 2;
  if(dist == 2) return 4;
  if(dist == 1) return 8;
  if(dist == 0) return 16;
  return 0;
}


__device__ __inline__
void device_clear_visited(GameBoard* this_board){
    for (int i=0; i<361; i++){
        this_board->visited[i] = 0;
    }
}

__device__ __inline__
void device_board_construct(GameBoard* this_board, int s){
    this_board->size = s;
    this_board->current_player_state = 1;
    for (int i=0; i<s; i++){
        for (int j=0; j<s; j++){
            this_board->draw[i*s+j] = 0;
            this_board->eval[i*s+j] = 0;
            this_board->classify[i*s+j] = 0;
        }
    }
}

__device__ __inline__
int device_board_addStone(GameBoard* this_board, int row, int col, int state){
    int s = this_board->size;
    if (row < 0 || row >= s || col < 0 || col >= s){
        return 0;
    }
    if (this_board->draw[row * s + col] != 0){
        return 0;
    }
    this_board->draw[row * s + col] = state;
    if (device_checkStone(this_board, row, col, state) == 0){
        this_board->draw[row * s + col] = 0;
        return 0;
    }
    device_board_get_terr(this_board);
    //board_printclassify(this_board);
    return 1;
}

__device__ __inline__
void device_delete_stone(GameBoard* this_board, int row, int col){
    int s = this_board->size;
    if (this_board->visited[row * s + col] == 1){
        return ;
    }
    this_board->visited[row * s + col] = 1;
    int state = this_board->draw[row * s + col];

    if (row > 0){
        if (this_board->draw[(row - 1) * s + col] == state){
            device_delete_stone(this_board, (row - 1), col);
            this_board->draw[(row - 1) * s + col] = 0;
        }
    } 

    if (row < s - 1){
        if (this_board->draw[(row + 1) * s + col] == state){
            device_delete_stone(this_board, (row + 1), col);
            this_board->draw[(row + 1) * s + col] = 0;
        }
    }

    if (col > 0){
        if (this_board->draw[row * s + col - 1] == state){
            device_delete_stone(this_board, row, col-1);
            this_board->draw[row * s + col - 1] = 0;
        }
    }

    if (col < s - 1){
        if (this_board->draw[row * s + col + 1] == state){
            device_delete_stone(this_board, row, col+1);
            this_board->draw[row * s + col + 1] = 0;
        }
    }
    this_board->draw[row * s + col] = 0;
}

__device__ __inline__
int device_get_liberties(GameBoard* this_board, int row, int col){
    int cnt = 0;
    int s = this_board->size;
    if (this_board->visited[row * s + col] == 1){
        return 0;
    }
    this_board->visited[row * s + col] = 1;
    int state = this_board->draw[row * s + col];

    if (row > 0){
        if (this_board->draw[(row - 1) * s + col] == state){
            cnt += device_get_liberties(this_board, (row - 1), col);
        } else {
            cnt += (this_board->draw[(row - 1) * s + col] == 0);
        }
    } 

    if (row < s - 1){
        if (this_board->draw[(row + 1) * s + col] == state){
            cnt += device_get_liberties(this_board, row + 1, col);
        } else {
            cnt += (this_board->draw[(row + 1) * s + col] == 0);
        }
    }

    if (col > 0){
        if (this_board->draw[row * s + col - 1] == state){
            cnt += device_get_liberties(this_board, row, col-1);
        } else {
            cnt += (this_board->draw[row * s + col - 1] == 0);
        }
    }

    if (col < s - 1){
        if (this_board->draw[row * s + col + 1] == state){
            cnt += device_get_liberties(this_board, row, col+1);
        } else {
            cnt += (this_board->draw[row * s + col + 1] == 0);
        }
    }
    return cnt;
}

__device__ __inline__
int device_checkStone(GameBoard* this_board, int row, int col, int state){

    int neighbors[4];
    int s = this_board->size;
    // Don't check outside the board
    if (row > 0) {neighbors[0] = (row - 1) * s + col;} else {neighbors[0] = -1;}
    if (row < s - 1) {neighbors[1] = (row + 1) * s + col;} else {neighbors[1] = -1;}
    if (col > 0) {neighbors[2] = row * s + col - 1;} else {neighbors[2] = -1;}
    if (col < s - 1) {neighbors[3] = row * s + col + 1;} else {neighbors[3] = -1;}

    int flag = 1;
    if (device_get_liberties(this_board, row, col) == 0){
        flag = 0;
    }

    int cur_row, cur_col;
    for (int idx = 0; idx < 4; idx++){
        if (neighbors[idx] != -1 && this_board->draw[neighbors[idx]] == -state){
            cur_row = neighbors[idx] / s;
            cur_col = neighbors[idx] % s;
            device_clear_visited(this_board);
            if (device_get_liberties(this_board, cur_row, cur_col) == 0){
                printf("delete\n");
                device_clear_visited(this_board);
                device_delete_stone(this_board, cur_row, cur_col);
                flag = 1;
            }
        }
    }
    return flag;
}

__device__ __inline__
void device_board_get_terr(GameBoard* this_board) {
    int s = this_board->size;

    for (int i=0; i< s*s; i++){
        this_board->eval[i] = 0;
    }

    int idx, dist, diff;
    for (int r = 0; r < s; r++){
        for (int c = 0; c < s; c++){
            idx = r * s + c;
            if (this_board->draw[idx] != 0){
                diff = this_board->draw[idx];
                for(int i = d_max_of(r - 4, 0); i < d_min_of(r + 5, s); i++){
                    for(int j = d_max_of(c - 4, 0); j < d_min_of(c + 5, s); j++){
                        dist = d_abso(r - i) + d_abso(c - j);
                        this_board->eval[i * s + j] += diff * d_mapping(dist);
                    }
                }
            }
        }
    }

    // dist = abso(-x - ii) + abso(y - jj);
    // this_board->board_eval[ii * 9 + jj] += (diff * mapping(dist) / 2);

    // dist = abso(x - ii) + abso(-y - jj);
    // this_board->board_eval[ii * 9 + jj] += (diff * mapping(dist) / 2);

    // dist = abso(16 - x - ii) + abso(y - jj);
    // this_board->board_eval[ii * 9 + jj] += (diff * mapping(dist) / 2);

    // dist = abso(x - ii) + abso(16 - y - jj);
    // this_board->board_eval[ii * 9 + jj] += (diff * mapping(dist) / 2);

    for(int i = 0; i < s * s; i++) {
        if(this_board->draw[i] == 1) {
            if(this_board->eval[i] <= 0) this_board->classify[i] = -1; //si le
            else this_board->classify[i] = 1;
        }
        else if(this_board->draw[i] == -1) {
            if(this_board->eval[i] >= 0) this_board->classify[i] = 1; //si le
            else this_board->classify[i] = -1;
        }
        else if(this_board->eval[i] > 0) this_board->classify[i] = 1;
        else if(this_board->eval[i] < 0) this_board->classify[i] = -1;
        else this_board->classify[i] = 0;
    }
}

__global__ void
kernel_monte_carlo(int* stones, int s, int* result){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (stones[index] == 0){
        GameBoard* next_board = new GameBoard;
        device_board_construct(next_board, s);
        for (int i=0; i<s; i++){
            for (int j=0; j<s; j++){
                if (stones[i*s+j] != 0){
                    if (stones[i*s+j] == 1){
                        device_board_addStone(next_board, i, j, 1);
                    } else {
                        device_board_addStone(next_board, i, j, -1);
                    }
                }
            }
        }
        int flag = device_board_addStone(next_board, index / s, index % s, -1);
        if (flag != 0){
            int w_count = 0;
            for (int i=0; i<s; i++){
                for (int j=0; j<s; j++){
                    if (next_board->classify[i*s+j] == 1){
                        w_count -= 1;
                    } else {
                        w_count += 1;
                    }
                }
            }
            result[index] = w_count;
        }
    }
}

int Monte_Carlo_Cuda(GameBoard* this_board) {
    int s = this_board->size;
    const int threadsPerBlock = 1;
    const int blocks = (s * s + threadsPerBlock - 1) / threadsPerBlock;

    int result[s*s];
    for(int i = 0; i < s * s; i++) {
        result[i] = -1;
    }

    int* device_stones;
    int* device_result; 

    cudaMalloc(&device_stones, s * s * sizeof(int));
    cudaMemcpy(device_stones, this_board->draw, 81 * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc(&device_result, s * s * sizeof(int));
    cudaMemcpy(device_result, result, s * s * sizeof(int), cudaMemcpyHostToDevice);

    // size_t limit = 0;
    // cudaDeviceGetLimit(&limit, cudaLimitStackSize);
    // printf("stack size = %u\n", (unsigned)limit);
    // limit = 65535;
    // cudaDeviceSetLimit(cudaLimitStackSize, limit);
    // cudaDeviceGetLimit(&limit, cudaLimitStackSize);
    // printf("stack size = %u\n", (unsigned)limit);

    kernel_monte_carlo<<<blocks, threadsPerBlock>>>(device_stones, s, device_result);

    cudaThreadSynchronize();
    cudaMemcpy(result, device_result, s * s * sizeof(int), cudaMemcpyDeviceToHost);

    int max_pos = rand() % (s * s);
    int max_val = -1;
    for (int i=0; i < s * s; i++){
        if (result[i] > max_val){
            max_val = result[i];
            max_pos = i;
        }
    }
    return max_pos;
}
