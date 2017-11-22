clc
clear

global A n linky ly lx visx visy slack
B = csvread('dataB.csv');
A = csvread('dataA.csv');

n = size(A,1);

tic

pA = A;
tinf = 9999999999.0;
linky = (-1) * ones(n,1);
ly = zeros(n,1);
lx = zeros(n,1);

for i = 1 : n
    lx(i) = -tinf;
    for j = 1 : n
        if (abs(A(i,j))-lx(i))>1e-10
            lx(i) = abs(A(i,j));
        end
    end
end
for x = 1 : n
    slack = tinf * ones(n,1);
    while 1
        visx = zeros(n,1);
        visy = zeros(n,1);
        if tofind(x) == 1
            break;
        end
        d = tinf;
        for i = 1 : n
            if visy(i)==0 && (d-slack(i))>1e-10
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

result = 0;
for i = 1 : n
    result = result + abs(A(i,linky(i)));
end

AA = zeros(n,n);
BB = zeros(n,1);
for i = 1 : n
    AA(linky(i),:) = pA(i,:);
    BB(linky(i),:) = B(i);
end

sA = sparse(AA);
setup.type = 'nofill';
[L,U] = ilu(sA,setup);
X = cgs(sA,BB,1e-6,10000,L,U);
toc

tic
ans = AA \ BB;
toc
        
        