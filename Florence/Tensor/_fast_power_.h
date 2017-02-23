#include <cmath>

inline double fast_pow(const double a, const int b) {
    if      (b==1)   return a;
    else if (b==2)   return a*a;
    else if (b==3)   return a*a*a;
    else if (b==4)   {double t0 = a*a; return t0*t0;}
    else if (b==5)   {double t0 = a*a; double t1=t0*t0; return t1*a;}
    else if (b==6)   {double t0 = a*a*a; return t0*t0;}
    else if (b==7)   {return a*fast_pow(a,6);}
    else if (b==8)   {double t0 = a*a; double t1=t0*t0; return t1*t1;}
    else if (b==9)   {double t0 = a*a*a; return t0*t0*t0;}
    else if (b==10)  {double t0 = fast_pow(a,5); return t0*t0;}
    else if (b==11)  {return a*a*fast_pow(a,9);}
    else if (b==12)  {double t0 = fast_pow(a,6); return t0*t0;}
    else if (b==13)  {return fast_pow(a,7)*fast_pow(a,6);}
    else if (b==14)  {double t0 = fast_pow(a,7); return t0*t0;}
    else if (b==15)  {return fast_pow(a,8)*fast_pow(a,7);}
    else if (b==16)  {double t0 = fast_pow(a,8); return t0*t0;}
    else if (b==17)  {return fast_pow(a,16)*a;}
    else if (b==18)  {double t0 = fast_pow(a,9); return t0*t0;}
    else if (b==19)  {return fast_pow(a,9)*fast_pow(a,10);}
    else if (b==20)  {double t0 = fast_pow(a,10); return t0*t0;}
    else if (b==21)  {return fast_pow(a,10)*fast_pow(a,11);}
    else if (b==22)  {return fast_pow(a,11)*fast_pow(a,11);}
    else if (b==23)  {return fast_pow(a,11)*fast_pow(a,12);}
    else if (b==24)  {double t0 = fast_pow(a,8); return t0*t0*t0;}
    else if (b==25)  {return fast_pow(fast_pow(a,5),5);}
    else if (b==26)  {return fast_pow(a,25)*a;}
    else if (b==27)  {double t0 = fast_pow(a,9); return fast_pow(t0,3);}
    else if (b==28)  {return fast_pow(a,27)*a;}
    else if (b==29)  {return fast_pow(a,27)*a*a;}
    else if (b==30)  {return fast_pow(a,27)*a*a*a;}
    else if (b==31)  {return fast_pow(a,27)*fast_pow(a,4);}
    else if (b==32)  {double t0 = fast_pow(a,16); return t0*t0;}
    else if (b==33)  {return fast_pow(a,32)*a;}
    else if (b==34)  {return fast_pow(a,32)*a*a;}
    else if (b==35)  {return fast_pow(a,32)*a*a*a;}
    else if (b==36)  {return fast_pow(fast_pow(a,6),6);}
    else if (b==37)  {return fast_pow(a,36)*a;}
    else if (b==38)  {return fast_pow(a,36)*a*a;}
    else if (b==39)  {return fast_pow(a,36)*a*a*a;}
    else if (b==40)  {double t0 = fast_pow(a,20); return t0*t0;}
    else if (b==48)  {double t0 = fast_pow(a,24); return t0*t0;}
    else if (b==49)  {return fast_pow(fast_pow(a,7),7);}
    else if (b==54)  {double t0 = fast_pow(a,27); return t0*t0;}
    else if (b==64)  {double t0 = fast_pow(a,32); return t0*t0;}
    else if (b==81)  {return fast_pow(fast_pow(a,9),9);}
    else if (b==100) {return fast_pow(fast_pow(a,10),10);}
    else if (b==121) {return fast_pow(fast_pow(a,11),11);}
    else if (b==128) {double t0 = fast_pow(a,64); return t0*t0;}
    else if (b==144) {return fast_pow(fast_pow(a,12),12);}
    else if (b==169) {return fast_pow(fast_pow(a,13),13);}
    else if (b==196) {return fast_pow(fast_pow(a,14),14);}
    else if (b==256) {double t0 = fast_pow(a,128); return t0*t0;}

    else if (b>40 and b<60) {
        double t0 = fast_pow(a,40);
        for (int i=40; i<b; ++i) {
            t0 *=a;
        }
        return t0;
    }
    else {
      return std::pow(a,b);
    }
}
