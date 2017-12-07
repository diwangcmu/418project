// #include <readfile.h>
// struct GameBoard{
// 	int size;
// 	int current_player_state; // 1 for black, -1 for white
// 	int draw[361];
// 	int eval[361];
// 	int classify[361];
// 	int visited[361];
// };
#include "readfile.h"

void clear_visited(GameBoard* this_board);
void board_printboard(GameBoard* this_board);
void delete_stone(GameBoard* this_board, int row, int col);
int get_liberties(GameBoard* this_board, int row, int col);
int checkStone(GameBoard* this_board, int row, int col, int state);
void board_get_terr(GameBoard* this_board);
void board_printclassify(GameBoard* this_board);
int board_monte_carlo(GameBoard* this_board);
int test();

int min_of(int x, int y) {
  if(x < y) return x;
  return y;
}

int max_of(int x, int y) {
  if(x > y) return x;
  return y;
}

int abso(int x) {
  if(x < 0) return -x;
  return x;
}

int mapping(int dist) {
  if(dist == 4) return 1;
  if(dist == 3) return 2;
  if(dist == 2) return 4;
  if(dist == 1) return 8;
  if(dist == 0) return 16;
  return 0;
}

void clear_visited(GameBoard* this_board){
	for (int i=0; i<361; i++){
		this_board->visited[i] = 0;
	}
}

void board_construct(GameBoard* this_board, int s){
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

void board_printboard(GameBoard* this_board){
	int s = this_board->size;
	for (int i = 0; i < s; i++){
		// printf("#%d",i+1);
		for (int j = 0; j < s; j++){
			if (this_board->draw[i*s+j] != 0){
				if (this_board->draw[i*s+j] == 1){
					printf(" x");
				} else {
					printf(" o");
				}
			} else {
				printf(" .");
			}
		}
		printf("\n");
	}
	for (int i=0; i<40; i++){
		printf("#");
	}
	printf("\n");
}

int board_addStone(GameBoard* this_board, int row, int col, int state){
	int s = this_board->size;
	if (row < 0 || row >= s || col < 0 || col >= s){
		return 0;
	}
	if (this_board->draw[row * s + col] != 0){
		return 0;
	}
	this_board->draw[row * s + col] = state;
	if (checkStone(this_board, row, col, state) == 0){
		this_board->draw[row * s + col] = 0;
		return 0;
	}
	board_get_terr(this_board);
	//board_printclassify(this_board);
	return 1;
}

void delete_stone(GameBoard* this_board, int row, int col){
	int s = this_board->size;
	if (this_board->visited[row * s + col] == 1){
		return ;
	}
	this_board->visited[row * s + col] = 1;
	int state = this_board->draw[row * s + col];

	if (row > 0){
		if (this_board->draw[(row - 1) * s + col] == state){
			delete_stone(this_board, (row - 1), col);
			this_board->draw[(row - 1) * s + col] = 0;
		}
	} 

	if (row < s - 1){
		if (this_board->draw[(row + 1) * s + col] == state){
			delete_stone(this_board, (row + 1), col);
			this_board->draw[(row + 1) * s + col] = 0;
		}
	}

	if (col > 0){
		if (this_board->draw[row * s + col - 1] == state){
			delete_stone(this_board, row, col-1);
			this_board->draw[row * s + col - 1] = 0;
		}
	}

	if (col < s - 1){
		if (this_board->draw[row * s + col + 1] == state){
			delete_stone(this_board, row, col+1);
			this_board->draw[row * s + col + 1] = 0;
		}
	}
	this_board->draw[row * s + col] = 0;
}

int get_liberties(GameBoard* this_board, int row, int col){
	int cnt = 0;
	int s = this_board->size;
	if (this_board->visited[row * s + col] == 1){
		return 0;
	}
	this_board->visited[row * s + col] = 1;
	int state = this_board->draw[row * s + col];

	if (row > 0){
		if (this_board->draw[(row - 1) * s + col] == state){
			cnt += get_liberties(this_board, (row - 1), col);
		} else {
			cnt += (this_board->draw[(row - 1) * s + col] == 0);
		}
	} 

	if (row < s - 1){
		if (this_board->draw[(row + 1) * s + col] == state){
			cnt += get_liberties(this_board, row + 1, col);
		} else {
			cnt += (this_board->draw[(row + 1) * s + col] == 0);
		}
	}

	if (col > 0){
		if (this_board->draw[row * s + col - 1] == state){
			cnt += get_liberties(this_board, row, col-1);
		} else {
			cnt += (this_board->draw[row * s + col - 1] == 0);
		}
	}

	if (col < s - 1){
		if (this_board->draw[row * s + col + 1] == state){
			cnt += get_liberties(this_board, row, col+1);
		} else {
			cnt += (this_board->draw[row * s + col + 1] == 0);
		}
	}
	return cnt;
}

int checkStone(GameBoard* this_board, int row, int col, int state){

	int neighbors[4];
	int s = this_board->size;
    // Don't check outside the board
    if (row > 0) {neighbors[0] = (row - 1) * s + col;} else {neighbors[0] = -1;}
    if (row < s - 1) {neighbors[1] = (row + 1) * s + col;} else {neighbors[1] = -1;}
    if (col > 0) {neighbors[2] = row * s + col - 1;} else {neighbors[2] = -1;}
    if (col < s - 1) {neighbors[3] = row * s + col + 1;} else {neighbors[3] = -1;}

	int flag = 1;
	if (get_liberties(this_board, row, col) == 0){
		flag = 0;
	}

	int cur_row, cur_col;
	for (int idx = 0; idx < 4; idx++){
		if (neighbors[idx] != -1 && this_board->draw[neighbors[idx]] == -state){
			cur_row = neighbors[idx] / s;
			cur_col = neighbors[idx] % s;
			clear_visited(this_board);
			if (get_liberties(this_board, cur_row, cur_col) == 0){
				printf("delete\n");
				clear_visited(this_board);
				delete_stone(this_board, cur_row, cur_col);
				flag = 1;
			}
		}
	}
	return flag;
}

int board_monte_carlo(GameBoard* this_board){
	int s = this_board->size;
	int max_pos = rand() % (s * s);
	int max_val = -1;

	for (int next_step = 0; next_step < s*s; next_step ++){
		if (this_board->draw[next_step] == 0){
			GameBoard* next_board = new GameBoard;
			board_construct(next_board, s);
			for (int i=0; i<s; i++){
				for (int j=0; j<s; j++){
					if (this_board->draw[i*s+j] != 0){
						if (this_board->draw[i*s+j] == 1){
							board_addStone(next_board, i, j, 1);
						} else {
							board_addStone(next_board, i, j, -1);
						}
					}
				}
			}

			int flag = board_addStone(next_board, next_step / s, next_step % s, -1);
			if (flag != 0){
				board_get_terr(next_board);

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

				// printf("(%d, %d)\n", next_step, w_count);

				if (w_count > max_val){
					max_val = w_count;
					max_pos = next_step;
				}
			}
		}
	}
	return max_pos;
}

void board_printclassify(GameBoard* this_board){
	int s = this_board->size;
	for (int i = 0; i < s; i++){
		for (int j = 0; j < s; j++){
			if (this_board->classify[i*s+j] != 0){
				if (this_board->classify[i*s+j] == 1){
					printf(" B");
				} else {
					printf(" W");
				}
			} else {
				printf(" .");
			}
		}
		printf("\n");
	}
	for (int i=0; i<100; i++){
		printf("#");
	}
	printf("\n");
}

void board_get_terr(GameBoard* this_board) {
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
				for(int i = max_of(r - 4, 0); i < min_of(r + 5, s); i++){
					for(int j = max_of(c - 4, 0); j < min_of(c + 5, s); j++){
						dist = abso(r - i) + abso(c - j);
						this_board->eval[i * s + j] += diff * mapping(dist);
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