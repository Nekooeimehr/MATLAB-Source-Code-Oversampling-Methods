function d=sample(I,P,N)
% This code is from the authors of the paper MWMOTE. The paper can be found
% in the following link:
% http://ieeexplore.ieee.org/xpls/abs_all.jsp?arnumber=6361394&tag=1


%samples N data from the input vectors according to probability
%distribution P;

[m,n]=size(P);
C=zeros(m,1);
prev=0;
for i=1:m
    C(i)=P(i)+prev;
    prev=C(i);
end
d=[];

for i=1:N
    rn=rand(1);
 
    for j=1:m
        if(rn<=C(j))
            d=[d;I(j,:)];
            break;
        end;
    end
end

