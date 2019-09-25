// #pragma once 
#include <iostream>
#include <math.h>


#define DIM 100 // Dataset size

//#define BASICS 
#define LEARN

#define RADIUS 0.5
#define BATCH_SIZE 10

double rule(double x, double y, double r=RADIUS); 
void printHello();
void demo(bool training=true);
