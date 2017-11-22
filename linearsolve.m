function [ X ] = linearsolve( A, b )

tic

[m,n]=size(A);
x0 = zeros(m,1);
r0 = b - A * x0;
r0x = r0;
j = 1;
maxiter = 2000;
norm0 = norm(r0);
tol = 0.000001;
while j<=maxiter
    for i = 1 : m
        fprintf('%f ',x0(i));
    end
    fprintf('\n');
    rou1 = dot(r0, r0x);
    disp(rou1);
    if rou1==0
        break;
    end
    if j==1
        p1 = r0; p0 = r0;
    else
        beta0 = rou1/rou0*alpha0/omega0;
        p1 = r0+beta0*(p0-omega0*v0);
    end
    px = p1;
    v1 = A*px;
    for i = 1 : m
        fprintf('%f ', v1(i));
    end
    alpha1 = rou1/dot(v1,r0x);
    disp(dot(v1,r0x));
    disp(alpha1);
    s = r0 - alpha1*v1;
    if norm(s) / norm0 < tol
        x1 = x0 + alpha1*px;
        break;
    end
    sx = s;
    t = A*sx;
    omega1 = dot(s,t)/dot(t,t);
    x1 = x0 + alpha1*px + omega1*sx;
    r1 = s - omega1*t;
    if norm(r1) / norm0 < tol
        break;
    end
    if omega1==0
        break;
    end
    rou0 = rou1; r0 = r1; p0 = p1; alpha0 = alpha1; omega0 = omega1; v0 = v1; x0 = x1;
    j=j+1;
end
%disp(j);

X = - x1;
toc
