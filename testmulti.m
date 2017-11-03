clc
clear

format long;

global A n linky ly lx visx visy slack
B = csvread('dataB.csv');
A = csvread('dataA.csv');
n = size(A,1);
pA = A;
tinf = 9999999999.0;
linky = (-1) * ones(n,1);
ly = zeros(n,1);
lx = zeros(n,1);

for i = 1 : n
    for j = 1 : n
        if abs(A(i,j))>1e-15
            A(i,j) = log10(abs(A(i,j)));
        else
            A(i,j) = -tinf;
        end
    end
end
        

for i = 1 : n
    lx(i) = -tinf;
    for j = 1 : n
        if (A(i,j)-lx(i))>1e-15
            lx(i) = A(i,j);
        end
    end
end
for x = 1 : n
    slack = tinf * ones(n,1);
    while 1
        visx = zeros(n,1);
        visy = zeros(n,1);
        if tofind2(x) == 1
            break;
        end
        d = tinf;
        for i = 1 : n
            if visy(i)==0 && (d-slack(i))>1e-15
                d = slack(i);
            end
        end
        for i = 1 : n
            if visx(i)==1
                lx(i) = lx(i) - d;
            end
        end
        for i = 1 : n
            if visy(i)==1   
                ly(i) = ly(i) + d;
            else
                slack(i) = slack(i) -d;
            end
        end
    end
end

AA = zeros(n,n);
BB = zeros(n,1);
for i = 1 : n
    AA(linky(i),:) = pA(i,:);
    BB(i,:) = B(linky(i));
end

sA = sparse(AA);
setup.type = 'nofill';
[L,U] = ilu(sA,setup);
X = bicgstab(sA,BB,1e-6,10000,L,U);
X = cgs(sA,BB,1e-6,10000);
X = linearsolve(sA,BB);
        
        
        
