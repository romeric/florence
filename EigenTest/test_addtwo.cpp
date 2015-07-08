#include <cy_cpp.h>
#include <iostream>

int main()
{
	double x=20.1;
	double y=5.3;

	double z; 
	z = addtwo(x,y);

	std::cout << z << std::endl;

	return 0;
}

// g++ test_addtwo.cpp -o test_addtwo