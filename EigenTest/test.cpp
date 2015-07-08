#include <iostream>
#include <Eigen/Dense>
using Eigen::MatrixXd;
#include <IGESControl_Reader.hxx>



int main()
{
  MatrixXd m(2,2);
  m(0,0) = 3;
  m(1,0) = 2.5;
  m(0,1) = -1;
  m(1,1) = m(1,0) + m(0,1);
  std::cout << m << std::endl;


  IGESControl_Reader reader; 
  IFSelect_ReturnStatus stat  = reader.ReadFile("filename.igs"); 
}
