#include "Stone.cpp"
#include "Chain.cpp"
#include <iostream>
#include "readfile.h"

void grid_construct(Grid* this_grid, int size, int player);
int grid_try_add(Grid* this_grid, int row, int col, int state);
int grid_addStone(Grid* this_grid, int row, int col, int state);
int grid_getLiberties(Grid* this_grid, Chain* cur_chain);
int grid_checkStone(Grid* this_grid, Stone* s);
int grid_search(Grid* this_grid, int range_count, stone_pos* range_stones[], int black_count, stone_pos* black_stones[]);
int grid_check_win(Grid* this_grid, int black_count, stone_pos* black_stones[]);
int grid_checklive(Grid* this_grid, int row, int col, int s);
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

int grid_search(Grid* this_grid, int range_count, stone_pos* range_stones[], int black_count, stone_pos* black_stones[]){
	if (grid_check_win(this_grid, black_count, black_stones) == 0){
		int global_flag, flag, result;
		if (this_grid->player == 1) global_flag = -1;
		if (this_grid->player == 0) global_flag = 1;

		for (int i=0; i<range_count; i++){
			stone_pos* cur = range_stones[i];
			if (this_grid->stones[cur->row][cur->col] == NULL){
				//printf("create new grid!!!\n");
				this_grid->next_grid[i] = new Grid;
				grid_construct(this_grid->next_grid[i], this_grid->size, this_grid->player_reverse);
				for (int r=0; r<9; r++){
					for (int c=0; c<9; c++){
						if (this_grid->stones[r][c] != NULL){
							grid_addStone(this_grid->next_grid[i], r, c, this_grid->stones[r][c]->state);
						}
					}
				}
				flag = grid_addStone(this_grid->next_grid[i], range_stones[i]->row, range_stones[i]->col, this_grid->player);
				if (flag == 1){
					result = grid_search(this_grid->next_grid[i], range_count, range_stones, black_count, black_stones);
					if (this_grid->player == 1 && result == 1){
						global_flag = 1;
					}
					if (this_grid->player == 0 && result == -1){
						global_flag = -1;
					}
				}
			}
		}
		return global_flag;
	} else {
		return grid_check_win(this_grid, black_count, black_stones);
	}
}

int grid_check_win(Grid* this_grid, int black_count, stone_pos* black_stones[]){
	//grid_printboard(this_grid);
	stone_pos* cur;
	int eye[9][9];

	for (int i=0; i<9; i++){
		for (int j=0; j<9; j++){
			eye[i][j] = 0;
		}
	}

	int row, col;

	int black_flag = 0;
	int eye_count = 0;
	for (int i=0;i<black_count;i++){
		cur = black_stones[i];
		row = cur->row; col = cur->col;
		if (this_grid->stones[row][col] != NULL){
			black_flag = 1;

	        //Don't check outside the board
	        if (row > 0) {
	        	if (grid_checklive(this_grid, row-1, col, 1) == 1 && eye[row-1][col] == 0){
	        		eye_count += 1;
	        		eye[row-1][col] = 1;
	        	}
	        }
	        if (row < this_grid->size - 1) {
	        	if (grid_checklive(this_grid, row+1, col, 1) == 1 && eye[row+1][col] == 0){
	        		eye_count += 1;
	        		eye[row+1][col] = 1;
	        	}
	        }
	        if (col > 0) {
	        	if (grid_checklive(this_grid, row, col-1, 1) == 1 && eye[row][col-1] == 0){
	        		eye_count += 1;
	        		eye[row][col-1] = 1;
	        	}
	        }
	        if (col < this_grid->size - 1) {
	        	if (grid_checklive(this_grid, row, col+1, 1) == 1 && eye[row][col+1] == 0){
	        		eye_count += 1;
	        		eye[row][col+1] = 1;
	        	}
	        }
		}
	}

	if (black_flag == 1){
		if (eye_count >= 2){
			//printf("Black stones alive!!!\n");	
			return 1;
		} else {
			//printf("Black stones unknown!!!\n");
		}
	} else {
		//printf("All black stones dead!!!\n");
		return -1;
	}
	return 0;
}

int grid_checklive(Grid* this_grid, int row, int col, int s){
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

void grid_printboard(Grid* this_grid){
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
