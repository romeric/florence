#include <cpp_backend.h>

int main()
{

    std::vector<int> ee(3,0);
    std::vector<std::vector<int> > e(10,ee);

    std::vector<double> pp(3,0);
    std::vector<std::vector<double> > p(10,pp);

//    std::vector<std::vector<double> > p;
//    std::vector<std::vector<int> > e;

//    int e,p,d;
    std::vector<std::vector<double> > d;

    d = manipulate(e,p);

    std::cout << "all good" << std::endl;

    return 0;
}
