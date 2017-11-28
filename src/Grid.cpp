#include "Stone.cpp"
#include "Chain.cpp"
#include <iostream>
#include "readfile.h"

class Grid{
	public:

	int size;
	Stone* stones[9][9];
	int player;
	int player_reverse;
	Grid* next_grid[81];

	void construct(int size, int player){
		this->size = size;
		if (player == 0){
			this->player = 0;
			this->player_reverse = 1;
		} else {
			this->player = 1;
			this->player_reverse = 0;
		}
		for (int i=0; i<9; i++){
			for (int j=0; j<9; j++){
				this->stones[i][j] = NULL;
			}
		}
	} 

	int try_add(int row, int col, int state){
		int s;
		if (state == 1){
			s = 0;
		} else {
			s = 1;
		}

		Stone* newStone= new Stone;
		newStone->construct(row, col, s);
		this->stones[row][col] = newStone;

		Stone* neighbors[4];
        for (int i=0;i<4;i++){
        	neighbors[i] = NULL;
        }

        // Don't check outside the board
        if (row > 0) {
            neighbors[0] = this->stones[row - 1][col];
        }
        if (row < this->size - 1) {
            neighbors[1] = this->stones[row + 1][col];
        }
        if (col > 0) {
            neighbors[2] = this->stones[row][col - 1];
        }
        if (col < this->size - 1) {
            neighbors[3] = this->stones[row][col + 1];
        }

        Stone* neighbor;

        int flag = 1;
        for (int i=0; i<4;i++){
        	if (neighbors[i] != NULL){
        		neighbor = neighbors[i];
        		if (neighbor->state != newStone->state){
        			if (getLiberties(neighbor->chain) == 0){
        				flag = 0;
        				break;
        			}
        		}
        	}
        }

        this->stones[row][col] = NULL;

        return flag;
	}

	int addStone(int row, int col, int state){
		Stone* newStone = new Stone;
		newStone->construct(row, col, state);
		this->stones[row][col] = newStone;

        Stone* neighbors[4];
        for (int i=0;i<4;i++){
        	neighbors[i] = NULL;
        }

        // Don't check outside the board
        if (row > 0) {
            neighbors[0] = this->stones[row - 1][col];
        }
        if (row < this->size - 1) {
            neighbors[1] = this->stones[row + 1][col];
        }
        if (col > 0) {
            neighbors[2] = this->stones[row][col - 1];
        }
        if (col < this->size - 1) {
            neighbors[3] = this->stones[row][col + 1];
        }

        Chain* current_chain = new Chain;
        current_chain->addStone(newStone);

        Stone* neighbor;

        int flag = 0;
        for (int i=0; i<4;i++){
        	if (neighbors[i] != NULL){
        		neighbor = neighbors[i];
        		if (neighbor->state != newStone->state){
        			if (checkStone(neighbor) == 1){
        				flag = 1;
        			}
        		} else {
        			if (neighbor->chain != newStone->chain){
        				neighbor->chain->join(newStone->chain);
        			}
        		}
        	}
        }

        if (getLiberties(newStone->chain) == 0 && flag == 0){
        	this->stones[row][col] = NULL;
        	return 0;
        }
        return 1;
	}

	int getLiberties(Chain* cur_chain){
		//int pos;
		int cnt = 0;

		for (int i=0; i<cur_chain->stones.size(); i++){
			Stone* s = cur_chain->stones[i];
			int row = s->row;
			int col = s->col;

			Stone* neighbors[4];
	        for (int j=0;j<4;j++){
	        	neighbors[j] = NULL;
	        }

	        // Don't check outside the board
	        if (row > 0) {
	            neighbors[0] = this->stones[row - 1][col];
	            if (neighbors[0] == NULL) cnt += 1;
	        }
	        if (row < this->size - 1) {
	            neighbors[1] = this->stones[row + 1][col];
	            if (neighbors[1] == NULL) cnt += 1;
	        }
	        if (col > 0) {
	            neighbors[2] = this->stones[row][col - 1];
	            if (neighbors[2] == NULL) cnt += 1;
	        }
	        if (col < this->size - 1) {
	            neighbors[3] = this->stones[row][col + 1];
	            if (neighbors[3] == NULL) cnt += 1;
	        }
		}
		return cnt;
	}

	int checkStone(Stone* s){
		int flag = 0;
		if (getLiberties(s->chain) == 0){
			flag = 1;
			for (int i=0; i<s->chain->stones.size(); i++){
				this->stones[s->chain->stones[i]->row][s->chain->stones[i]->col] = NULL;
			}
		}
		return flag;
	}

    //new
    int search_first(int range_count, stone_pos* range_stones[], int black_count, stone_pos* black_stones[]) {
        int* feedback = malloc(sizeof(int));
        int result;
        searchCuda(this, range_count, range_stones, black_count, black_stones, feedback);
        result = *feedback;
        free(feedback);
        return result;
    }

	int search(int range_count, stone_pos* range_stones[], int black_count, stone_pos* black_stones[]){
		if (this->check_win(black_count, black_stones) == 0){
			int global_flag, flag, result;
			if (this->player == 1) global_flag = -1;
			if (this->player == 0) global_flag = 1;

			for (int i=0; i<range_count; i++){
				stone_pos* cur = range_stones[i];
				if (this->stones[cur->row][cur->col] == NULL){
					//printf("create new grid!!!\n");
					this->next_grid[i] = new Grid;
					this->next_grid[i]->construct(this->size, this->player_reverse);
					for (int r=0; r<9; r++){
						for (int c=0; c<9; c++){
							if (this->stones[r][c] != NULL){
								this->next_grid[i]->addStone(r, c, this->stones[r][c]->state);
							}
						}
					}
					flag = this->next_grid[i]->addStone(range_stones[i]->row, range_stones[i]->col, this->player);
					if (flag == 1){
						result = this->next_grid[i]->search(range_count, range_stones, black_count, black_stones);
						if (this->player == 1 && result == 1){
							global_flag = 1;
							//this->next_grid[i]->printboard();
						}
						if (this->player == 0 && result == -1){
							global_flag = -1;
							//this->next_grid[i]->printboard();
						}
					}
				}
			}
			return global_flag;
		} else {
			return this->check_win(black_count, black_stones);
		}
	}

	int check_win(int black_count, stone_pos* black_stones[]){
		//this->printboard();
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
			if (this->stones[row][col] != NULL){
				black_flag = 1;

		        //Don't check outside the board
		        if (row > 0) {
		        	if (this->checklive(row-1, col, 1) == 1 && eye[row-1][col] == 0){
		        		eye_count += 1;
		        		eye[row-1][col] = 1;
		        	}
		        }
		        if (row < this->size - 1) {
		        	if (this->checklive(row+1, col, 1) == 1 && eye[row+1][col] == 0){
		        		eye_count += 1;
		        		eye[row+1][col] = 1;
		        	}
		        }
		        if (col > 0) {
		        	if (this->checklive(row, col-1, 1) == 1 && eye[row][col-1] == 0){
		        		eye_count += 1;
		        		eye[row][col-1] = 1;
		        	}
		        }
		        if (col < this->size - 1) {
		        	if (this->checklive(row, col+1, 1) == 1 && eye[row][col+1] == 0){
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

	int checklive(int row, int col, int s){
		if (this->stones[row][col] != NULL){
			return 0;
		}
		Stone* neighbors[4];
        for (int j=0;j<4;j++){
        	neighbors[j] = NULL;
        }

        int cnt, flag;
        flag = 1;

        // Don't check outside the board
        if (row > 0) {
            neighbors[0] = this->stones[row - 1][col];
            if (neighbors[0] == NULL || neighbors[0]->state != s) flag = 0;
        }
        if (row < this->size - 1) {
            neighbors[1] = this->stones[row + 1][col];
            if (neighbors[1] == NULL || neighbors[1]->state != s) flag = 0;
        }
        if (col > 0) {
            neighbors[2] = this->stones[row][col - 1];
            if (neighbors[2] == NULL || neighbors[2]->state != s) flag = 0;
        }
        if (col < this->size - 1) {
            neighbors[3] = this->stones[row][col + 1];
            if (neighbors[3] == NULL || neighbors[3]->state != s) flag = 0;
        }

        if (flag == 1){
        	return this->try_add(row, col, s);
        }

        return 0;
	}

	void printboard(){
		for (int i = 0; i < this->size; i++){
			printf("#");
			for (int j = 0; j < this->size; j++){
				if (this->stones[i][j] != NULL){
					if (this->stones[i][j]->state == 1){
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
};