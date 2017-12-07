#include "Stone.cpp"
#include "Chain.cpp"
#include <iostream>
#include "readfile.h"

struct Grid{
	int size;
	Stone* stones[9][9];
	int player;
	int player_reverse;
	Grid* next_grid[81];
};

void grid_construct(Grid* this_grid, int size, int player);
int grid_try_add(Grid* this_grid, int row, int col, int state);
int grid_addStone(Grid* this_grid, int row, int col, int state);
int grid_getLiberties(Grid* this_grid, Chain* cur_chain);
int grid_checkStone(Grid* this_grid, Stone* s);
void grid_printboard(Grid* this_grid);

void grid_construct(Grid* this_grid, int size, int player){
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

int grid_try_add(Grid* this_grid, int row, int col, int state){
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

int grid_addStone(Grid* this_grid, int row, int col, int state){
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

int grid_getLiberties(Grid* this_grid, Chain* cur_chain){
	int cnt = 0;

	for (int i=0; i<cur_chain->chain_size; i++){
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

int grid_checkStone(Grid* this_grid, Stone* s){
	int flag = 0;
	if (grid_getLiberties(this_grid, s->chain) == 0){
		flag = 1;
		for (int i=0; i<s->chain->chain_size; i++){
			this_grid->stones[s->chain->stones[i]->row][s->chain->stones[i]->col] = NULL;
		}
	}
	return flag;
}

void grid_printboard(Grid* this_grid){
	for (int i=0; i<22; i++){
		printf("#");
	}
    printf("\n");
    printf("# ");
    for (int i=1; i<10; i++){
        printf(" %d", i);
    }
    printf(" #\n");
	for (int i = 0; i < this_grid->size; i++){
		printf("#%d",i+1);
		for (int j = 0; j < this_grid->size; j++){
			if (this_grid->stones[i][j] != NULL){
				if (this_grid->stones[i][j]->state == 1){
					printf(" x");
				} else {
					printf(" o");
				}
			} else {
				printf(" .");
			}
		}
		printf(" #\n");
	}
	for (int i=0; i<22; i++){
		printf("#");
	}
	printf("\n");
}
