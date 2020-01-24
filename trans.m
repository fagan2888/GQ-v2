function [y]=transx(x,tcode)
%    Transform x
%    Return Series with same dimension and corresponding dates
%    Missing values where not calculated
%    -- Tcodes:
%             0 First Difference
%             1 Log-Level
%             2 Log-First-Difference
%             3 Log-Second-Difference


%  Translated from the Gauss procs of Stock&Watson(2005),'Implications of
%  dynamic factor models for VAR analysis'
%  Dimitris Korobilis, June 2007

small=1.0e-06;
relvarm=.00000075;
relvarq=.000625;     %HP parameter
                     %.00000075 for monthly data
                     %.000625 for quarterly data, see Harvey/Jeager (1993), page 234 @
n=size(x,1);
y=zeros(n,1);        %storage space for y

if tcode == 0
    y = diff(x);
	y(1,:) = [];
    
elseif tcode == 1
    y = log(x);
	y(1,:) = [];
    
elseif tcode == 2
    y = diff(log(x));
	y(1,:) = [];
    
elseif tcode == 3
    y = diff(diff(log(x)));
	y(1,:) = [];
    y(2,:) = [];
end