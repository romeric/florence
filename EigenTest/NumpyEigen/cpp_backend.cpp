
#include <cpp_backend.h>


std::vector<std::vector<double> > manipulate(std::vector<std::vector<int> > elements_std,std::vector<std::vector<double> > points_std)
{
    Eigen::MatrixXi elements = Eigen::MatrixXi::Zero(elements_std.size(),elements_std[0].size());
    Eigen::MatrixXd points = Eigen::MatrixXd::Zero(points_std.size(),points_std[0].size());

    for (int i=0; i < elements_std.size(); ++i )
    {
        for (int j=0; j<elements_std[0].size();++j)
        {
            elements(i,j) = elements_std[i][j];
        }
    }

    for (int i=0; i < points_std.size(); ++i )
    {
        for (int j=0; j<points_std[0].size();++j)
        {
            points(i,j) = points_std[i][j];
        }
    }

    elements_std.clear(); points_std.clear();




    std::vector<std::vector<double> > dirichletbc;
    dirichletbc.clear();
    for (int i=0; i < elements.rows(); ++i )
    {
        std::vector<double> dummy_3(2,(double)i*6.2);
        dirichletbc.push_back(dummy_3);
    }

    //std::cout << "Hello from C++!" << std::endl;

    return dirichletbc;
}
