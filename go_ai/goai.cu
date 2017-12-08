#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include "readfile.h"

struct GameBoard;
void host_clear_visited(GameBoard* this_board);
void host_board_construct(GameBoard* this_board, int s);
int host_board_addStone(GameBoard* this_board, int row, int col, int state);
void host_delete_stone(GameBoard* this_board, int row, int col);
int host_get_liberties(GameBoard* this_board, int row, int col);
int host_checkStone(GameBoard* this_board, int row, int col, int state);

void host_clear_visited(GameBoard* this_board){
    for (int i=0; i<361; i++){
        this_board->visited[i] = 0;
    }
}

void host_board_construct(GameBoard* this_board, int s){
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

int host_board_addStone(GameBoard* this_board, int row, int col, int state){
    int s = this_board->size;
    if (row < 0 || row >= s || col < 0 || col >= s){
        return 0;
    }
    if (this_board->draw[row * s + col] != 0){
        return 0;
    }
    this_board->draw[row * s + col] = state;
    if (host_checkStone(this_board, row, col, state) == 0){
        this_board->draw[row * s + col] = 0;
        return 0;
    }
    //device_board_get_terr(this_board);
    //board_printclassify(this_board);
    return 1;
}

void host_delete_stone(GameBoard* this_board, int row, int col){
    int s = this_board->size;
    if (this_board->visited[row * s + col] == 1){
        return ;
    }
    this_board->visited[row * s + col] = 1;
    int state = this_board->draw[row * s + col];

    if (row > 0){
        if (this_board->draw[(row - 1) * s + col] == state){
            host_delete_stone(this_board, (row - 1), col);
            this_board->draw[(row - 1) * s + col] = 0;
        }
    } 

    if (row < s - 1){
        if (this_board->draw[(row + 1) * s + col] == state){
            host_delete_stone(this_board, (row + 1), col);
            this_board->draw[(row + 1) * s + col] = 0;
        }
    }

    if (col > 0){
        if (this_board->draw[row * s + col - 1] == state){
            host_delete_stone(this_board, row, col-1);
            this_board->draw[row * s + col - 1] = 0;
        }
    }

    if (col < s - 1){
        if (this_board->draw[row * s + col + 1] == state){
            host_delete_stone(this_board, row, col+1);
            this_board->draw[row * s + col + 1] = 0;
        }
    }
    this_board->draw[row * s + col] = 0;
}

int host_get_liberties(GameBoard* this_board, int row, int col){
    int cnt = 0;
    int s = this_board->size;
    if (this_board->visited[row * s + col] == 1){
        return 0;
    }
    this_board->visited[row * s + col] = 1;
    int state = this_board->draw[row * s + col];

    if (row > 0){
        if (this_board->draw[(row - 1) * s + col] == state){
            cnt += host_get_liberties(this_board, (row - 1), col);
        } else {
            cnt += (this_board->draw[(row - 1) * s + col] == 0);
        }
    } 

    if (row < s - 1){
        if (this_board->draw[(row + 1) * s + col] == state){
            cnt += host_get_liberties(this_board, row + 1, col);
        } else {
            cnt += (this_board->draw[(row + 1) * s + col] == 0);
        }
    }

    if (col > 0){
        if (this_board->draw[row * s + col - 1] == state){
            cnt += host_get_liberties(this_board, row, col-1);
        } else {
            cnt += (this_board->draw[row * s + col - 1] == 0);
        }
    }

    if (col < s - 1){
        if (this_board->draw[row * s + col + 1] == state){
            cnt += host_get_liberties(this_board, row, col+1);
        } else {
            cnt += (this_board->draw[row * s + col + 1] == 0);
        }
    }
    return cnt;
}

int host_checkStone(GameBoard* this_board, int row, int col, int state){

    int neighbors[4];
    int s = this_board->size;
    // Don't check outside the board
    if (row > 0) {neighbors[0] = (row - 1) * s + col;} else {neighbors[0] = -1;}
    if (row < s - 1) {neighbors[1] = (row + 1) * s + col;} else {neighbors[1] = -1;}
    if (col > 0) {neighbors[2] = row * s + col - 1;} else {neighbors[2] = -1;}
    if (col < s - 1) {neighbors[3] = row * s + col + 1;} else {neighbors[3] = -1;}

    int flag = 1;
    if (host_get_liberties(this_board, row, col) == 0){
        flag = 0;
    }

    int cur_row, cur_col;
    for (int idx = 0; idx < 4; idx++){
        if (neighbors[idx] != -1 && this_board->draw[neighbors[idx]] == -state){
            cur_row = neighbors[idx] / s;
            cur_col = neighbors[idx] % s;
            host_clear_visited(this_board);
            if (host_get_liberties(this_board, cur_row, cur_col) == 0){
                printf("delete\n");
                host_clear_visited(this_board);
                host_delete_stone(this_board, cur_row, cur_col);
                flag = 1;
            }
        }
    }
    return flag;
}

__global__ void
kernel_monte_carlo(int* stones, int s, int* result){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (stones[index * s * s] != -2){
        int eval[361];
        for (int i=0; i< s*s; i++){
            eval[i] = 0;
        }

        int idx, dist, diff;
        //calculate eval
        for (int r = 0; r < s; r++){
            for (int c = 0; c < s; c++){
                idx = r * s + c;
                if (stones[idx + index * s * s] != 0){
                    diff = stones[idx + index * s * s];

                    int i1,i2,j1,j2;

                    if (r-4 > 0){i1 = r-4;} else {i1=0;}
                    if (r+5 < s){i2 = r+5;} else {i2=s;}
                    if (c-4 > 0){j1 = c-4;} else {j1=0;}
                    if (c+5 < s){j2 = c+5;} else {j2=s;}

                    for(int i = i1; i < i2; i++){
                        for(int j = j1; j < j2; j++){

                            int ab1, ab2,m;

                            if (r-i > 0){ ab1 = r-i;}
                            else {ab1 = i-r;}
                            if (c-j > 0){ ab2 = c-j;}
                            else {ab2 = j-c;}

                            if(dist == 4) m = 1;
                            if(dist == 3) m = 2;
                            if(dist == 2) m = 4;
                            if(dist == 1) m = 8;
                            if(dist == 0) m = 16;
                            if(dist > 4)  m = 0;

                            eval[i * s + j] += diff * m;
                        }
                    }
                }
            }
        }

        int w_count = 0;
        for(int i = 0; i < s * s; i++) {
            if(stones[i + index * s * s] == 1) {
                if(eval[i] < 0) w_count += 1;
                else w_count -= 1;
            }
            else if(stones[i + index * s * s] == -1) {
                if(eval[i] > 0) w_count -= 1;
                else w_count += 1;
            }
            else if(eval[i] > 0) w_count -= 1;
            else if(eval[i] < 0) w_count += 1;
        }
        result[index] = w_count;
    }
}

int Monte_Carlo_Cuda(GameBoard* this_board, int n) {
    int s = this_board->size;
    int num = s * s * s * s;

    const int threadsPerBlock = 128;
    const int blocks = (num + threadsPerBlock - 1) / threadsPerBlock;

    int result[num];
    for(int i = 0; i < num; i++) {
        result[i] = -100;
    }

    int stones[num * s * s];
    int move_seq[num * n];

    //generating moving sequences

    for (int i=0; i<num * n; i++){
        move_seq[i] = 0;
    }

    int p;
    for (int i=1; i<num; i++){
        for (int j=i*n; j<(i+1)*n; j++){
            move_seq[j] = move_seq[j-n];
        }
        p = (i+1) * n - 1;
        while (move_seq[p] == s*s-1){
            move_seq[p] = 0;
            p -= 1;
        }
        move_seq[p] += 1;
    }

    for (int idx = 0; idx < num; idx ++){
        GameBoard* next_board = new GameBoard;
        host_board_construct(next_board, s);
        for (int r = 0; r < s; r++){
            for (int c = 0; c < s; c++){
                next_board->draw[r * s + c] = this_board->draw[r * s + c];
            }
        }

        int flag = 1;
        int type = 1;
        int cur_flag;
        for (int k=0; k<n; k++){
            type *= (-1);
            cur_flag = host_board_addStone(next_board, move_seq[idx * n + k] / s, move_seq[idx * n + k] % s, -1);
            if (cur_flag == 0){
                flag = 0;
                break;
            }
        }

        if (flag == 1){
            for (int i = 0; i < s * s; i++){
                stones[idx * s * s + i] = next_board->draw[i];
            }
        } else {
            stones[idx * s * s] = -2;
        }
    }

    int* device_stones;
    int* device_result; 

    cudaMalloc(&device_stones, num * s * s * sizeof(int));
    cudaMemcpy(device_stones, stones, num * s * s * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc(&device_result, num * sizeof(int));
    cudaMemcpy(device_result, result, num * sizeof(int), cudaMemcpyHostToDevice);

    // size_t limit = 0;
    // cudaDeviceGetLimit(&limit, cudaLimitStackSize);
    // printf("stack size = %u\n", (unsigned)limit);
    // limit = 65535;
    // cudaDeviceSetLimit(cudaLimitStackSize, limit);
    // cudaDeviceGetLimit(&limit, cudaLimitStackSize);
    // printf("stack size = %u\n", (unsigned)limit);

    kernel_monte_carlo<<<blocks, threadsPerBlock>>>(device_stones, s, device_result);

    cudaThreadSynchronize();

    int max_pos = rand() % (s * s);
    float max_val = -101.0;

    cudaMemcpy(result, device_result, num * sizeof(int), cudaMemcpyDeviceToHost);
    // for (int i=0; i < num; i++){
    //     printf("(%d, %d) -> %d\n", move_seq[i * n], move_seq[i*n+1], result[i]);
    // }
    // printf("\n");

    int partial_num = s * s;
    for (int idx = 0; idx < s*s; idx++){
        int cnt = 0;
        int sum = 0;
        for (int i=0; i < partial_num; i++){
            if (result[idx*partial_num + i] != -100){
                cnt += 1;
                sum += result[idx * partial_num + i];
            }
        }

        if ((float(sum)/cnt) > max_val){
            max_val = (float(sum)/cnt);
            max_pos = idx;
        }
    }

    cudaFree(result);
    cudaFree(device_stones);
    cudaFree(device_result);

    return max_pos;
}
