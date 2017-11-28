#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <getopt.h>
#include <string>
#include "GameBoard.cpp"


using namespace std;

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
    board->construct(9, black, blacks, white, whites, range, ranges);

    if (board->search() == -1){
        printf("blacks can be killed!\n");
    } else {
        printf("sha bu si\n");
    }

    return 0;
}
