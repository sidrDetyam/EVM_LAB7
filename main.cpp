
#include <iostream>
#include <chrono>
#include <vector>
#include "SquareMatrix.h"

using namespace std;


void test(){
    SquareMatrix<4> m;
    float c = 1;
    for(size_t i=0; i<4; ++i){
        for(size_t j=0; j<4; ++j){
            m[i][j] = c;
            ++c;
        }
    }
    m[1][1] = 7;
    m[1][2] = 7;
    m[1][3] = 7;
    m[3][3] = 17;

    cout << endl;
    cout << m << endl;
    cout << m.inverseMatrix(100) << endl;
}


void test2(){

    SquareMatrix<1000> m = move(SquareMatrix<1000>::idMatrix());
    m.inverseMatrix(1);
}


int main(){

    auto t1 = chrono::high_resolution_clock::now();
    test2();
    auto t2 = chrono::high_resolution_clock::now();
    cout << (float)chrono::duration_cast<chrono::microseconds>( t2 - t1 ).count() / 1000000;

    return 0;
}