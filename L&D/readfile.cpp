#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <getopt.h>
#include <string>
#include "readfile.h"
#include "GameBoard.cpp"
using namespace std;

// sequential compile: g++ -std=c++11 -O2 readfile.cpp -o seq -lm
// sequential run: ./seq -s test.txt

// parallel compile: make
// parallel run: ./cudaSearch -s test.txt

int searchCuda(Grid* curGrid, int range_count, stone_pos** range_stones, int black_count, stone_pos** black_stones);

void usage(const char* progname) {
    printf("Usage: %s [options]\n", progname);
    printf("Program Options:\n");
    printf("  -s  --url              Text file to read\n");
    printf("  -?  --help             This message\n");
}

int main(int argc, char** argv)
{
    ifstream inFile;

    // parse commandline options ////////////////////////////////////////////
    int opt;
    static struct option long_options[] = {
        {"arraysize",  1, 0, 's'},
        {"help",       0, 0, '?'},
        {0 ,0, 0, 0}
    };

    while ((opt = getopt_long(argc, argv, "?s:", long_options, NULL)) != EOF) {

        switch (opt) {
        case 's':
            // printf("%s\n",optarg);
            inFile.open(optarg);
            break;
        case '?':
        default:
            usage(argv[0]);
            return 1;
        }
    }
    // end parsing of commandline options //////////////////////////////////////

    if(!inFile) {
        cerr << "Unable to open file\n";
        exit(1);
    }
    int black, white, range;
    inFile >> black;
    inFile >> white;
    inFile >> range;
    stone_pos* blacks[black];
    stone_pos* whites[white];
    stone_pos* ranges[range];

    for(int i = 0; i < black; i++) {
        blacks[i] = new stone_pos;
        inFile >> blacks[i]->row;
        inFile >> blacks[i]->col;
    }

    for(int i = 0; i < white; i++) {
        whites[i] = new stone_pos;
        inFile >> whites[i]->row;
        inFile >> whites[i]->col;
    }

    for(int i = 0; i < range; i++) {
        ranges[i] = new stone_pos;
        inFile >> ranges[i]->row;
        inFile >> ranges[i]->col;
    }

    GameBoard* board = new GameBoard;
    board_construct_host(board, 9, black, blacks, white, whites, range, ranges);
    grid_printboard_host(board->initial_grid);

    int flag = searchCuda(board->initial_grid, range, ranges, black, blacks);
    //int flag = grid_search(board->initial_grid, board->range_count, board->range_stones, board->black_count, board->black_stones);
    
    if (flag == -1){
        printf("blacks can be killed!\n");
    } else {
        printf("sha bu si\n");
    }

    return 0;
}
