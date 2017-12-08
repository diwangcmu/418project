//#include "go_Decision.h"
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <getopt.h>
#include <iostream>
#include <string>
#include "GameBoard.cpp"
#include "readfile.h"
using namespace std;

int Monte_Carlo_Cuda(GameBoard* this_board, int n);

//g++ "-I/System/Library/Frameworks/JavaVM.framework/Versions/Current/Headers" \
"-I/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX10.12.sdk/System/Library/Frameworks/JavaVM.framework/Versions/A/Headers" \
-c go_Decision.cpp; g++ -dynamiclib -o libgo_Decision.jnilib go_Decision.o; java -Djava.library.path="." go.Main

// no-interface version
int main(int argc, char** argv)
{
    int size = 9;
    GameBoard* board = new GameBoard;
    board_construct(board, size);

    int row, col, next_move;
    cin >> row;
    while (row != -1){
        //printf("aaaaa%d\n", test());
        cin >> col;
        board_addStone(board, row, col, 1);
        next_move = Monte_Carlo_Cuda(board, 2);
        //next_move = board_monte_carlo(board);
        printf("add white stone %d\n", next_move);
        while (board_addStone(board, next_move / size, next_move % size, -1) == 0){
            next_move = rand() % (size * size);
        }
        board_printboard(board);
        cin >> row;
    }
    return 0;
}



// int local_size = 0;
// GameBoard* board = new GameBoard;

// JNIEXPORT jboolean JNICALL Java_go_Decision_start
//   (JNIEnv *env, jobject obj, jint size) {
//     local_size = size;
//     board_construct(board, size);
//     return true;
// }

// int next_move = -1;

// JNIEXPORT jint JNICALL Java_go_Decision_getResponseMove
//   (JNIEnv *env, jobject obj, jint move) {
//     int row = move / local_size;
//     int col = move % local_size;
//     board_addStone(board, row, col, 1);

//     next_move = board_monte_carlo(board);
//     if (board_addStone(board, next_move % local_size, next_move / local_size, -1) == 0){
//         next_move = rand() % (local_size * local_size);
//     }
//     return next_move;
// }

