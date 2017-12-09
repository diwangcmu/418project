#include "readfile.h"
#include "Grid.cpp"

void board_construct_host(GameBoard* this_board, int s, int black, stone_pos* blacks[], int white, stone_pos* whites[], int range, stone_pos* ranges[]);
void grid_construct_host(Grid* this_grid, int size, int player);
int grid_addStone_host(Grid* this_grid, int row, int col, int state);
int grid_getLiberties_host(Grid* this_grid, Chain* cur_chain);
int grid_checkStone_host(Grid* this_grid, Stone* s);
void stone_construct_host(Stone* this_stone, int row, int col, int state);
void chain_construct_host(Chain* this_chain);
void chain_addStone_host(Chain* this_chain, Stone* s);
void chain_join_host(Chain* this_chain, Chain* chain);
void grid_printboard_host(Grid* this_grid);

void board_construct_host(GameBoard* this_board, int s, int black, stone_pos* blacks[], int white, stone_pos* whites[], int range, stone_pos* ranges[]){
	this_board->size = s;
	this_board->initial_grid = new Grid;
	grid_construct_host(this_board->initial_grid, s, 0);
	this_board->current_player_state = 0;

	this_board->black_count = black;
	for (int i=0; i<black; i++){
		grid_addStone_host(this_board->initial_grid, blacks[i]->row, blacks[i]->col, 1);
		this_board->black_stones[i] = blacks[i];
	}

	this_board->white_count = white;
	for (int i=0; i<white; i++){
		grid_addStone_host(this_board->initial_grid, whites[i]->row, whites[i]->col, 0);
		this_board->white_stones[i] = whites[i];
	}

	this_board->range_count = range;
	for (int i=0; i<range; i++){
		this_board->range_stones[i] = ranges[i];
	}
}

void grid_construct_host(Grid* this_grid, int size, int player){
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

int grid_addStone_host(Grid* this_grid, int row, int col, int state){
	Stone* newStone = new Stone;
	stone_construct_host(newStone, row, col, state);
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
    chain_construct_host(current_chain);
    chain_addStone_host(current_chain, newStone);

    Stone* neighbor;

    int flag = 0;
    for (int i=0; i<4;i++){
    	if (neighbors[i] != NULL){
    		neighbor = neighbors[i];
    		if (neighbor->state != newStone->state){
    			if (grid_checkStone_host(this_grid, neighbor) == 1){
    				flag = 1;
    			}
    		} else {
    			if (neighbor->chain != newStone->chain){
    				chain_join_host(neighbor->chain, newStone->chain);
    			}
    		}
    	}
    }

    if (grid_getLiberties_host(this_grid, newStone->chain) == 0 && flag == 0){
    	this_grid->stones[row][col] = NULL;
    	return 0;
    }
    return 1;
}

int grid_getLiberties_host(Grid* this_grid, Chain* cur_chain){
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

int grid_checkStone_host(Grid* this_grid, Stone* s){
	int flag = 0;
	if (grid_getLiberties_host(this_grid, s->chain) == 0){
		flag = 1;
		for (int i=0; i<s->chain->chain_size; i++){
			this_grid->stones[s->chain->stones[i]->row][s->chain->stones[i]->col] = NULL;
		}
	}
	return flag;
}

void grid_printboard_host(Grid* this_grid){
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

void stone_construct_host(Stone* this_stone, int row, int col, int state){
	this_stone->state = state;
	this_stone->row = row;
	this_stone->col = col;
	this_stone->chain = NULL;
}

void chain_construct_host(Chain* this_chain){
	
	this_chain->chain_size = 0;
}

void chain_addStone_host(Chain* this_chain, Stone* s){
	s->chain = this_chain;
	this_chain->stones[this_chain->chain_size] = s;
	this_chain->chain_size += 1;
}

void chain_join_host(Chain* this_chain, Chain* chain){
	for (int i = 0; i != chain->chain_size; ++i){
		chain_addStone_host(this_chain, chain->stones[i]);
	}
}

// int board_search(GameBoard* this_board){
// 	grid_printboard(this_board->initial_grid);
// 	return grid_search(this_board->initial_grid, this_board->range_count, this_board->range_stones, this_board->black_count, this_board->black_stones);
// }