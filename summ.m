function [sums] = summ(A)
	sums=0.;
	for i=[1:size(A,1)]
		for j=[1:size(A,2)]
			sums = sums + A(i,j);
		end
	end 

