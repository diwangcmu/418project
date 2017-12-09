#include <iostream>
#include "readfile.h"
struct Chain;

void stone_construct(Stone* this_stone, int row, int col, int state){
	this_stone->state = state;
	this_stone->row = row;
	this_stone->col = col;
	this_stone->chain = NULL;
}