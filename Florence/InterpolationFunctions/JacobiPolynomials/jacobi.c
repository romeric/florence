#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void jacobi(const unsigned short n, const double xi, const double a, const double b, double *P)
{
	double a1n,a2n,a3n,a4n;
	unsigned short p;

	P[0]=1.0;
	if (n>0)
	{
		P[1] = 0.5*((a-b)+(a+b+2)*xi);
	}
	if (n>1)
	{

		for (p=1; p<n; p++)
		{
			a1n = 2*(p+1)*(p+a+b+1)*(2*p+a+b);
			a2n = (2*p+a+b+1)*(a*a-b*b);
			a3n = (2*p+a+b)*(2*p+a+b+1)*(2*p+a+b+2);
			a4n = 2*(p+a)*(p+b)*(2*p+a+b+2);

			P[p+1] = ((a2n+a3n*xi)*P[p]-a4n*P[p-1])/a1n;
		}
	}
}

void diffjacobi(const unsigned short n, const double xi, const double a, const double b, const unsigned short opt, double *dP)
{
	unsigned short p;
	double *P = malloc( (n+1)*sizeof(double));
    if (opt==1)
	{
		jacobi(n,xi,a+1,b+1,P);
	}
	else
	{
		jacobi(n,xi,a,b,P);
	}

	for (p=1; p<n+1;p++)
	{
		dP[p] = 0.5*(a+b+p+1)*P[p-1];
	}

	free(P);
}

/* int main() */
/* { */
/* 	unsigned short n = 10; */
/* 	double xi = 0.5; */
/* 	double a=0.0; */
/* 	double b=0.0; */
/* 	unsigned short opt = 0; */
/* 	double *P = malloc( (n+1)*sizeof(double)); */
/* 	double *dP = malloc( (n+1)*sizeof(double)); */
/*  */
/* 	clock_t begin, end; */
/* 	double time_spent; */
/*  */
/* 	#<{(| jacobi(n,xi,a,b,P); |)}># */
/* 	diffjacobi(n,xi,a,b,opt,dP); */
/* 	int i= n+1; */
/* 	for (i=0;i<n+1; i++) */
/* 		printf("%6.6f ",dP[i]); */
/* 	printf("\n"); */
/*  */
/* 	begin = clock(); */
/* 	end = clock(); */
/* 	time_spent = (double)(end - begin) / CLOCKS_PER_SEC; */
/* 	#<{(| printf("time elapsed was: %6.6f\n", time_spent); |)}># */
/*  */
/* 	free(P); */
/* 	free(dP); */
/* 	return 0; */
/* } */
