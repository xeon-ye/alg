function [ result ] = tofind2( x )
global A n linky ly lx visx visy slack

visx(x) = 1;
for y = 1 : n
    if visy(y)==1
        continue;
    end
    t = lx(x)+ly(y)-A(x,y);
    if abs(t)<=1e-15
        visy(y) = 1;
        if linky(y)==-1 || tofind2(linky(y))==1
            linky(y) = x;
            result = 1;
            return;
        end
    else
        if (slack(y)-t)>1e-15
            slack(y) = t;
        end
    end
end
result = 0;
            

            

end

