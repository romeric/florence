#include <cy_cpp.h>

double addtwo(double x, double y)
{
	double z;
	z = x+y;
	return z; 
}




// g++ cy_cpp.cpp -I. -c -fPIC
// g++ -shared -o libcy_cpp.so cy_cpp.o