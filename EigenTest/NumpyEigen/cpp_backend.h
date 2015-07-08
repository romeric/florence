#ifndef CPP_BACKEND_H
#define CPP_BACKEND_H

#include <Eigen/Core>
#include <Eigen/Dense>
#include <vector>
#include <iostream>

std::vector<std::vector<double> > manipulate(std::vector<std::vector<int> > elements_std,std::vector<std::vector<double> > points_std);
//std::vector<double> manipulate(std::vector<int> elements_std,std::vector<double> points_std);
//int manipulate(int a,int b);

#endif // CPP_BACKEND_H

