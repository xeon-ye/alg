package zju.lfp.utils;


/**
 * Created by IntelliJ IDEA.
 * User: hhh
 * Date: 2005-12-12
 * Time: 10:27:27
 */
public class MatrixOperation {
    public MatrixOperation() {
    }

    // 1 QR分解
    //  运用HouseHolder对一般的M×N矩阵进行QR变换
    //  A 双精度二维实型数组  返回时 其右上三角矩阵存放上三角矩阵R
    //  B 双精度二维实型数组   返回时 存放正交矩阵Q
    // 如果系数矩阵 列线性相关 则运算失败返回值为零
    public static int QRDecomposition(double[] a, int m, int n, double[] q) {
        int i, j, k, l, nn, p, jj;
        double u, alpha, w, t;
        if (m < n) return 0;
        for (i = 0; i <= m - 1; i++)
            for (j = 0; j <= m - 1; j++) {
                l = i * m + j;
                q[l] = 0.0;
                if (i == j) q[l] = 1.0;
            }
        nn = n;
        if (m == n) nn = m - 1;
        for (k = 0; k <= nn - 1; k++) {
            u = 0.0;
            l = k * n + k;
            for (i = k; i <= m - 1; i++) {
                w = Math.abs(a[i * n + k]);
                if (w > u) u = w;
            }
            alpha = 0.0;
            for (i = k; i <= m - 1; i++) {
                t = a[i * n + k] / u;
                alpha = alpha + t * t;
            }
            if (a[l] > 0.0) u = -u;
            alpha = u * Math.sqrt(alpha);
            if (Math.abs(alpha) + 1.0 == 1.0) return (0);
            u = Math.sqrt(2.0 * alpha * (alpha - a[l]));
            if ((u + 1.0) != 1.0) {
                a[l] = (a[l] - alpha) / u;
                for (i = k + 1; i <= m - 1; i++) {
                    p = i * n + k;
                    a[p] = a[p] / u;
                }
                for (j = 0; j <= m - 1; j++) {
                    t = 0.0;
                    for (jj = k; jj <= m - 1; jj++)
                        t = t + a[jj * n + k] * q[jj * m + j];
                    for (i = k; i <= m - 1; i++) {
                        p = i * m + j;
                        q[p] = q[p] - 2.0 * t * a[i * n + k];
                    }
                }
                for (j = k + 1; j <= n - 1; j++) {
                    t = 0.0;
                    for (jj = k; jj <= m - 1; jj++)
                        t = t + a[jj * n + k] * a[jj * n + j];
                    for (i = k; i <= m - 1; i++) {
                        p = i * n + j;
                        a[p] = a[p] - 2.0 * t * a[i * n + k];
                    }
                }
                a[l] = alpha;
                for (i = k + 1; i <= m - 1; i++)
                    a[i * n + k] = 0.0;
            }
        }
        for (i = 0; i <= m - 2; i++)
            for (j = i + 1; j <= m - 1; j++) {
                p = i * m + j;
                l = j * m + i;
                t = q[p];
                q[p] = q[l];
                q[l] = t;
            }
        return (1);
    }

    /// 2. 解最小二乘问题
    /// ＝＝＝＝运用HouseHolder变换对一般m*n矩阵进行QR分解,在此基础上解最小二乘问题
    /// A,双精度二维实型数组，体积为m*n；返回时，其右上三角存放上三角矩阵R
    /// m,n 行数，列数
    /// b，长度为m，存放右端常数向量。返回时，存放方程组的最小二乘解.
    /// q, 双精度二维实型数组，体积为m*m。返回时存放正交矩阵Q
    /// 如系数矩阵 列线性相关，则运算会失败, 返回值为0。
    public static int SolveLmsByQR(double[] a, int m, int n, double[] b, double[] q) {
        int i, j;
        double d;
        if (QRDecomposition(a, m, n, q) == 0) return (0);
        double[] c = new double[n];
        for (i = 0; i <= n - 1; i++) {
            d = 0.0;
            for (j = 0; j <= m - 1; j++)
                d = d + q[j * m + i] * b[j];
            c[i] = d;
        }
        b[n - 1] = c[n - 1] / a[n * n - 1];
        for (i = n - 2; i >= 0; i--) {
            d = 0.0;
            for (j = i + 1; j <= n - 1; j++)
                d = d + a[i * n + j] * b[j];
            b[i] = (c[i] - d) / a[i * n + i];
        }
        //free(c);
        return (1);
    }

    ///==========================================================================
    /// 3. 求实系数方程的所有根
    /// ======== 将问题转化为一个上hessenberg矩阵特征值问题，在此基础上得到方程全部根
    ///  如果返回值小于0，说明迭代到最大次数仍没有满足精度要求
    ///   a: a[n],a[n-1],...,a[0],长度为n+1
    ///   n: 方程阶数
    ///   xr,xi: n个根实部、虚部
    ///   eps: 误差要求
    ///   jt:  最大迭代次数
    public static int GetEquationRootsByQR(double[] a, int n, double[] xr, double[] xi, double eps, int jt) {
        int i, j;
        double[] q = new double[n * n];
        for (j = 0; j <= n - 1; j++)
            q[j] = -a[n - j - 1] / a[n];
        for (j = n; j <= n * n - 1; j++)
            q[j] = 0.0;
        for (i = 0; i <= n - 2; i++)
            q[(i + 1) * n + i] = 1.0;
        i = GetUpperHessenBergMatrixEigenvalueByQR(q, n, xr, xi, eps, jt);
        //delete[] q;
        return (i);
    }

    ///==========================================================================
    ///  4. 得到上HessenBerg矩阵的特征根
    ///  ======= 采用带原点位移的双重步QR方法
    ///  如果返回值小于0，说明迭代到最大次数仍没有满足精度要求
    ///   a: n*n,存放上H矩阵A
    ///   n: 方程阶数
    ///   u,v: n个根实部、虚部
    ///   eps: 误差要求
    ///   jt:  最大迭代次数
    public static int GetUpperHessenBergMatrixEigenvalueByQR(double[] a, int n, double[] u, double[] v, double eps, int jt) {
        int m, it, i, j, k, l, ii, jj, kk, ll;
        double b, c, w, g, xy, p, q, r, x, s, e, f, z, y;
        it = 0;
        m = n;
        while (m != 0) {
            l = m - 1;
            while ((l > 0) && (Math.abs(a[l * n + l - 1]) > eps *
                    (Math.abs(a[(l - 1) * n + l - 1]) + Math.abs(a[l * n + l])))) l = l - 1;
            ii = (m - 1) * n + m - 1;
            jj = (m - 1) * n + m - 2;
            kk = (m - 2) * n + m - 1;
            ll = (m - 2) * n + m - 2;
            if (l == m - 1) {
                u[m - 1] = a[(m - 1) * n + m - 1];
                v[m - 1] = 0.0;
                m = m - 1;
                it = 0;
            } else if (l == m - 2) {
                b = -(a[ii] + a[ll]);
                c = a[ii] * a[ll] - a[jj] * a[kk];
                w = b * b - 4.0 * c;
                y = Math.sqrt(Math.abs(w));
                if (w > 0.0) {
                    xy = 1.0;
                    if (b < 0.0) xy = -1.0;
                    u[m - 1] = (-b - xy * y) / 2.0;
                    u[m - 2] = c / u[m - 1];
                    v[m - 1] = 0.0;
                    v[m - 2] = 0.0;
                } else {
                    u[m - 1] = -b / 2.0;
                    u[m - 2] = u[m - 1];
                    v[m - 1] = y / 2.0;
                    v[m - 2] = -v[m - 1];
                }
                m = m - 2;
                it = 0;
            } else {
                if (it >= jt) {
                    //printf("fail\n");
                    return (-1);
                }
                it = it + 1;
                for (j = l + 2; j <= m - 1; j++)
                    a[j * n + j - 2] = 0.0;
                for (j = l + 3; j <= m - 1; j++)
                    a[j * n + j - 3] = 0.0;
                for (k = l; k <= m - 2; k++) {
                    if (k != l) {
                        p = a[k * n + k - 1];
                        q = a[(k + 1) * n + k - 1];
                        r = 0.0;
                        if (k != m - 2) r = a[(k + 2) * n + k - 1];
                    } else {
                        x = a[ii] + a[ll];
                        y = a[ll] * a[ii] - a[kk] * a[jj];
                        ii = l * n + l;
                        jj = l * n + l + 1;
                        kk = (l + 1) * n + l;
                        ll = (l + 1) * n + l + 1;
                        p = a[ii] * (a[ii] - x) + a[jj] * a[kk] + y;
                        q = a[kk] * (a[ii] + a[ll] - x);
                        r = a[kk] * a[(l + 2) * n + l + 1];
                    }
                    if ((Math.abs(p) + Math.abs(q) + Math.abs(r)) != 0.0) {
                        xy = 1.0;
                        if (p < 0.0) xy = -1.0;
                        s = xy * Math.sqrt(p * p + q * q + r * r);
                        if (k != l) a[k * n + k - 1] = -s;
                        e = -q / s;
                        f = -r / s;
                        x = -p / s;
                        y = -x - f * r / (p + s);
                        g = e * r / (p + s);
                        z = -x - e * q / (p + s);
                        for (j = k; j <= m - 1; j++) {
                            ii = k * n + j;
                            jj = (k + 1) * n + j;
                            p = x * a[ii] + e * a[jj];
                            q = e * a[ii] + y * a[jj];
                            r = f * a[ii] + g * a[jj];
                            if (k != m - 2) {
                                kk = (k + 2) * n + j;
                                p = p + f * a[kk];
                                q = q + g * a[kk];
                                r = r + z * a[kk];
                                a[kk] = r;
                            }
                            a[jj] = q;
                            a[ii] = p;
                        }
                        j = k + 3;
                        if (j >= m - 1) j = m - 1;
                        for (i = l; i <= j; i++) {
                            ii = i * n + k;
                            jj = i * n + k + 1;
                            p = x * a[ii] + e * a[jj];
                            q = e * a[ii] + y * a[jj];
                            r = f * a[ii] + g * a[jj];
                            if (k != m - 2) {
                                kk = i * n + k + 2;
                                p = p + f * a[kk];
                                q = q + g * a[kk];
                                r = r + z * a[kk];
                                a[kk] = r;
                            }
                            a[jj] = q;
                            a[ii] = p;
                        }
                    }
                }
            }
        }
        return (1);
    }

    ///==========================================================================
    /// 5. 用householder变换及变形QR算法对一般实矩阵进行奇异值分解
    /// 返回值小于0，表示出现了迭代60次还未求得某个奇异值的情况，此时，矩阵分解
    ///  式为UAV； 返回标志值大于0，正常返回
    /// a, m*n矩阵
    /// m，行数
    /// n, 列数
    /// u, m*m, 存放左奇异向量U
    /// v, n*n, 存放右奇异向量V
    /// eps, 给定精度要求
    /// ka,  整型变量，其值为max(m,n)+1
    public static int SinglarityValueDecompositionByQR(double[] a, int m, int n, double[] u, double[] v, double eps, int ka) {
        int i, j, k, l, it, ll, kk, ix, iy, mm, nn, iz, m1, ks;
        double d, dd, t, sm, sm1, em1, sk, ek, b, c, shh;
        double[] fg = new double[2];
        double[] cs = new double[2];
        double[] s = new double[ka];
        double[] e = new double[ka];
        double[] w = new double[ka];
        it = 60;
        k = n;
        if (m - 1 < n) k = m - 1;
        l = m;
        if (n - 2 < m) l = n - 2;
        if (l < 0) l = 0;
        ll = k;
        if (l > k) ll = l;
        if (ll >= 1) {
            for (kk = 1; kk <= ll; kk++) {
                if (kk <= k) {
                    d = 0.0;
                    for (i = kk; i <= m; i++) {
                        ix = (i - 1) * n + kk - 1;
                        d = d + a[ix] * a[ix];
                    }
                    s[kk - 1] = Math.sqrt(d);
                    if (s[kk - 1] != 0.0) {
                        ix = (kk - 1) * n + kk - 1;
                        if (a[ix] != 0.0) {
                            s[kk - 1] = Math.abs(s[kk - 1]);
                            if (a[ix] < 0.0) s[kk - 1] = -s[kk - 1];
                        }
                        for (i = kk; i <= m; i++) {
                            iy = (i - 1) * n + kk - 1;
                            a[iy] = a[iy] / s[kk - 1];
                        }
                        a[ix] = 1.0 + a[ix];
                    }
                    s[kk - 1] = -s[kk - 1];
                }
                if (n >= kk + 1) {
                    for (j = kk + 1; j <= n; j++) {
                        if ((kk <= k) && (s[kk - 1] != 0.0)) {
                            d = 0.0;
                            for (i = kk; i <= m; i++) {
                                ix = (i - 1) * n + kk - 1;
                                iy = (i - 1) * n + j - 1;
                                d = d + a[ix] * a[iy];
                            }
                            d = -d / a[(kk - 1) * n + kk - 1];
                            for (i = kk; i <= m; i++) {
                                ix = (i - 1) * n + j - 1;
                                iy = (i - 1) * n + kk - 1;
                                a[ix] = a[ix] + d * a[iy];
                            }
                        }
                        e[j - 1] = a[(kk - 1) * n + j - 1];
                    }
                }
                if (kk <= k) {
                    for (i = kk; i <= m; i++) {
                        ix = (i - 1) * m + kk - 1;
                        iy = (i - 1) * n + kk - 1;
                        u[ix] = a[iy];
                    }
                }
                if (kk <= l) {
                    d = 0.0;
                    for (i = kk + 1; i <= n; i++)
                        d = d + e[i - 1] * e[i - 1];
                    e[kk - 1] = Math.sqrt(d);
                    if (e[kk - 1] != 0.0) {
                        if (e[kk] != 0.0) {
                            e[kk - 1] = Math.abs(e[kk - 1]);
                            if (e[kk] < 0.0) e[kk - 1] = -e[kk - 1];
                        }
                        for (i = kk + 1; i <= n; i++)
                            e[i - 1] = e[i - 1] / e[kk - 1];
                        e[kk] = 1.0 + e[kk];
                    }
                    e[kk - 1] = -e[kk - 1];
                    if ((kk + 1 <= m) && (e[kk - 1] != 0.0)) {
                        for (i = kk + 1; i <= m; i++)
                            w[i - 1] = 0.0;
                        for (j = kk + 1; j <= n; j++)
                            for (i = kk + 1; i <= m; i++)
                                w[i - 1] = w[i - 1] + e[j - 1] * a[(i - 1) * n + j - 1];
                        for (j = kk + 1; j <= n; j++)
                            for (i = kk + 1; i <= m; i++) {
                                ix = (i - 1) * n + j - 1;
                                a[ix] = a[ix] - w[i - 1] * e[j - 1] / e[kk];
                            }
                    }
                    for (i = kk + 1; i <= n; i++)
                        v[(i - 1) * n + kk - 1] = e[i - 1];
                }
            }
        }
        mm = n;
        if (m + 1 < n) mm = m + 1;
        if (k < n) s[k] = a[k * n + k];
        if (m < mm) s[mm - 1] = 0.0;
        if (l + 1 < mm) e[l] = a[l * n + mm - 1];
        e[mm - 1] = 0.0;
        nn = m;
        if (m > n) nn = n;
        if (nn >= k + 1) {
            for (j = k + 1; j <= nn; j++) {
                for (i = 1; i <= m; i++)
                    u[(i - 1) * m + j - 1] = 0.0;
                u[(j - 1) * m + j - 1] = 1.0;
            }
        }
        if (k >= 1) {
            for (ll = 1; ll <= k; ll++) {
                kk = k - ll + 1;
                iz = (kk - 1) * m + kk - 1;
                if (s[kk - 1] != 0.0) {
                    if (nn >= kk + 1)
                        for (j = kk + 1; j <= nn; j++) {
                            d = 0.0;
                            for (i = kk; i <= m; i++) {
                                ix = (i - 1) * m + kk - 1;
                                iy = (i - 1) * m + j - 1;
                                d = d + u[ix] * u[iy] / u[iz];
                            }
                            d = -d;
                            for (i = kk; i <= m; i++) {
                                ix = (i - 1) * m + j - 1;
                                iy = (i - 1) * m + kk - 1;
                                u[ix] = u[ix] + d * u[iy];
                            }
                        }
                    for (i = kk; i <= m; i++) {
                        ix = (i - 1) * m + kk - 1;
                        u[ix] = -u[ix];
                    }
                    u[iz] = 1.0 + u[iz];
                    if (kk - 1 >= 1)
                        for (i = 1; i <= kk - 1; i++)
                            u[(i - 1) * m + kk - 1] = 0.0;
                } else {
                    for (i = 1; i <= m; i++)
                        u[(i - 1) * m + kk - 1] = 0.0;
                    u[(kk - 1) * m + kk - 1] = 1.0;
                }
            }
        }
        for (ll = 1; ll <= n; ll++) {
            kk = n - ll + 1;
            iz = kk * n + kk - 1;
            if ((kk <= l) && (e[kk - 1] != 0.0)) {
                for (j = kk + 1; j <= n; j++) {
                    d = 0.0;
                    for (i = kk + 1; i <= n; i++) {
                        ix = (i - 1) * n + kk - 1;
                        iy = (i - 1) * n + j - 1;
                        d = d + v[ix] * v[iy] / v[iz];
                    }
                    d = -d;
                    for (i = kk + 1; i <= n; i++) {
                        ix = (i - 1) * n + j - 1;
                        iy = (i - 1) * n + kk - 1;
                        v[ix] = v[ix] + d * v[iy];
                    }
                }
            }
            for (i = 1; i <= n; i++)
                v[(i - 1) * n + kk - 1] = 0.0;
            v[iz - n] = 1.0;
        }
        for (i = 1; i <= m; i++)
            for (j = 1; j <= n; j++)
                a[(i - 1) * n + j - 1] = 0.0;
        m1 = mm;
        it = 60;
        while (1 == 1) {
            if (mm == 0) {
                ppp(a, e, s, v, m, n);
                //free(s); //free(e); //free(w);
                return (1);
            }
            if (it == 0) {
                ppp(a, e, s, v, m, n);
                //free(s); //free(e); //free(w);
                return (-1);
            }
            kk = mm - 1;
            while ((kk != 0) && (Math.abs(e[kk - 1]) != 0.0)) {
                d = Math.abs(s[kk - 1]) + Math.abs(s[kk]);
                dd = Math.abs(e[kk - 1]);
                if (dd > eps * d) kk = kk - 1;
                else
                    e[kk - 1] = 0.0;
            }
            if (kk == mm - 1) {
                kk = kk + 1;
                if (s[kk - 1] < 0.0) {
                    s[kk - 1] = -s[kk - 1];
                    for (i = 1; i <= n; i++) {
                        ix = (i - 1) * n + kk - 1;
                        v[ix] = -v[ix];
                    }
                }
                while ((kk != m1) && (s[kk - 1] < s[kk])) {
                    d = s[kk - 1];
                    s[kk - 1] = s[kk];
                    s[kk] = d;
                    if (kk < n)
                        for (i = 1; i <= n; i++) {
                            ix = (i - 1) * n + kk - 1;
                            iy = (i - 1) * n + kk;
                            d = v[ix];
                            v[ix] = v[iy];
                            v[iy] = d;
                        }
                    if (kk < m)
                        for (i = 1; i <= m; i++) {
                            ix = (i - 1) * m + kk - 1;
                            iy = (i - 1) * m + kk;
                            d = u[ix];
                            u[ix] = u[iy];
                            u[iy] = d;
                        }
                    kk = kk + 1;
                }
                it = 60;
                mm = mm - 1;
            } else {
                ks = mm;
                while ((ks > kk) && (Math.abs(s[ks - 1]) != 0.0)) {
                    d = 0.0;
                    if (ks != mm) d = d + Math.abs(e[ks - 1]);
                    if (ks != kk + 1) d = d + Math.abs(e[ks - 2]);
                    dd = Math.abs(s[ks - 1]);
                    if (dd > eps * d) ks = ks - 1;
                    else
                        s[ks - 1] = 0.0;
                }
                if (ks == kk) {
                    kk = kk + 1;
                    d = Math.abs(s[mm - 1]);
                    t = Math.abs(s[mm - 2]);
                    if (t > d) d = t;
                    t = Math.abs(e[mm - 2]);
                    if (t > d) d = t;
                    t = Math.abs(s[kk - 1]);
                    if (t > d) d = t;
                    t = Math.abs(e[kk - 1]);
                    if (t > d) d = t;
                    sm = s[mm - 1] / d;
                    sm1 = s[mm - 2] / d;
                    em1 = e[mm - 2] / d;
                    sk = s[kk - 1] / d;
                    ek = e[kk - 1] / d;
                    b = ((sm1 + sm) * (sm1 - sm) + em1 * em1) / 2.0;
                    c = sm * em1;
                    c = c * c;
                    shh = 0.0;
                    if ((b != 0.0) || (c != 0.0)) {
                        shh = Math.sqrt(b * b + c);
                        if (b < 0.0) shh = -shh;
                        shh = c / (b + shh);
                    }
                    fg[0] = (sk + sm) * (sk - sm) - shh;
                    fg[1] = sk * ek;
                    for (i = kk; i <= mm - 1; i++) {
                        sss(fg, cs);
                        if (i != kk) e[i - 2] = fg[0];
                        fg[0] = cs[0] * s[i - 1] + cs[1] * e[i - 1];
                        e[i - 1] = cs[0] * e[i - 1] - cs[1] * s[i - 1];
                        fg[1] = cs[1] * s[i];
                        s[i] = cs[0] * s[i];
                        if ((cs[0] != 1.0) || (cs[1] != 0.0))
                            for (j = 1; j <= n; j++) {
                                ix = (j - 1) * n + i - 1;
                                iy = (j - 1) * n + i;
                                d = cs[0] * v[ix] + cs[1] * v[iy];
                                v[iy] = -cs[1] * v[ix] + cs[0] * v[iy];
                                v[ix] = d;
                            }
                        sss(fg, cs);
                        s[i - 1] = fg[0];
                        fg[0] = cs[0] * e[i - 1] + cs[1] * s[i];
                        s[i] = -cs[1] * e[i - 1] + cs[0] * s[i];
                        fg[1] = cs[1] * e[i];
                        e[i] = cs[0] * e[i];
                        if (i < m)
                            if ((cs[0] != 1.0) || (cs[1] != 0.0))
                                for (j = 1; j <= m; j++) {
                                    ix = (j - 1) * m + i - 1;
                                    iy = (j - 1) * m + i;
                                    d = cs[0] * u[ix] + cs[1] * u[iy];
                                    u[iy] = -cs[1] * u[ix] + cs[0] * u[iy];
                                    u[ix] = d;
                                }
                    }
                    e[mm - 2] = fg[0];
                    it = it - 1;
                } else if (ks == mm) {
                    kk = kk + 1;
                    fg[1] = e[mm - 2];
                    e[mm - 2] = 0.0;
                    for (ll = kk; ll <= mm - 1; ll++) {
                        i = mm + kk - ll - 1;
                        fg[0] = s[i - 1];
                        sss(fg, cs);
                        s[i - 1] = fg[0];
                        if (i != kk) {
                            fg[1] = -cs[1] * e[i - 2];
                            e[i - 2] = cs[0] * e[i - 2];
                        }
                        if ((cs[0] != 1.0) || (cs[1] != 0.0))
                            for (j = 1; j <= n; j++) {
                                ix = (j - 1) * n + i - 1;
                                iy = (j - 1) * n + mm - 1;
                                d = cs[0] * v[ix] + cs[1] * v[iy];
                                v[iy] = -cs[1] * v[ix] + cs[0] * v[iy];
                                v[ix] = d;
                            }
                    }
                } else {
                    kk = ks + 1;
                    fg[1] = e[kk - 2];
                    e[kk - 2] = 0.0;
                    for (i = kk; i <= mm; i++) {
                        fg[0] = s[i - 1];
                        sss(fg, cs);
                        s[i - 1] = fg[0];
                        fg[1] = -cs[1] * e[i - 1];
                        e[i - 1] = cs[0] * e[i - 1];
                        if ((cs[0] != 1.0) || (cs[1] != 0.0))
                            for (j = 1; j <= m; j++) {
                                ix = (j - 1) * m + i - 1;
                                iy = (j - 1) * m + kk - 2;
                                d = cs[0] * u[ix] + cs[1] * u[iy];
                                u[iy] = -cs[1] * u[ix] + cs[0] * u[iy];
                                u[ix] = d;
                            }
                    }
                }
            }
        }
        //return(1);
    }

    static void ppp(double[] a, double[] e, double[] s, double[] v, int m, int n) {
        int i, j, p, q;
        double d;
        if (m >= n) i = n;
        else
            i = m;
        for (j = 1; j <= i - 1; j++) {
            a[(j - 1) * n + j - 1] = s[j - 1];
            a[(j - 1) * n + j] = e[j - 1];
        }
        a[(i - 1) * n + i - 1] = s[i - 1];
        if (m < n) a[(i - 1) * n + i] = e[i - 1];
        for (i = 1; i <= n - 1; i++)
            for (j = i + 1; j <= n; j++) {
                p = (i - 1) * n + j - 1;
                q = (j - 1) * n + i - 1;
                d = v[p];
                v[p] = v[q];
                v[q] = d;
            }
    }

    static void sss(double[] fg, double[] cs) {
        double r, d;
        if ((Math.abs(fg[0]) + Math.abs(fg[1])) == 0.0) {
            cs[0] = 1.0;
            cs[1] = 0.0;
            d = 0.0;
        } else {
            d = Math.sqrt(fg[0] * fg[0] + fg[1] * fg[1]);
            if (Math.abs(fg[0]) > Math.abs(fg[1])) {
                d = Math.abs(d);
                if (fg[0] < 0.0) d = -d;
            }
            if (Math.abs(fg[1]) >= Math.abs(fg[0])) {
                d = Math.abs(d);
                if (fg[1] < 0.0) d = -d;
            }
            cs[0] = fg[0] / d;
            cs[1] = fg[1] / d;
        }
        r = 1.0;
        if (Math.abs(fg[0]) > Math.abs(fg[1])) r = cs[1];
        else if (cs[0] != 0.0) r = 1.0 / cs[0];
        fg[0] = d;
        fg[1] = r;
    }

    ///==========================================================================
    /// 6. 利用奇异值分解求一般m*n实矩阵A的广义逆
    /// 返回值小于0，表示在奇异值分解中出现了迭代60次还未满足精度要求；返回标志值大于0，正常返回
    /// a, m*n矩阵
    /// m，行数
    /// n, 列数
    /// aa, n*m, 存放广义逆
    /// u, m*m, 存放左奇异向量U
    /// v, n*n, 存放右奇异向量V
    /// eps, 奇异值分解时精度要求
    /// ka,  整型变量，其值为max(m,n)+1
    public static int GetGeneralizedInverseBySinglarityValueDecompositon(double[] a, int m, int n, double[] aa,
                                                                         double eps, double[] u, double[] v, int ka) {
        int i, j, k, l, t, p, q, f;
        i = SinglarityValueDecompositionByQR(a, m, n, u, v, eps, ka);
        if (i < 0) return (-1);
        j = n;
        if (m < n) j = m;
        j = j - 1;
        k = 0;
        while ((k <= j) && (a[k * n + k] != 0.0)) k = k + 1;
        k = k - 1;
        for (i = 0; i <= n - 1; i++)
            for (j = 0; j <= m - 1; j++) {
                t = i * m + j;
                aa[t] = 0.0;
                for (l = 0; l <= k; l++) {
                    f = l * n + i;
                    p = j * m + l;
                    q = l * n + l;
                    aa[t] = aa[t] + v[f] * u[p] / a[q];
                }
            }
        return (1);
    }

    ///==========================================================================
    /// 7. 求解线性最小二乘问题的广义逆法 ,1.14
    /// 返回值小于0，表示在奇异值分解中出现了迭代60次还未满足精度要求；返回标志值大于0，正常返回
    /// a, m*n矩阵
    /// m，行数
    /// n, 列数
    /// b, 一维向量，长度为m. 存放超定方程组右端常数向量
    /// x, 一维向量，长度为n。 存放超定方程组最小二乘解
    /// aa, n*m, 存放广义逆
    /// u, m*m, 存放左奇异向量U
    /// v, n*n, 存放右奇异向量V
    /// eps, 奇异值分解时精度要求
    /// ka,  整型变量，其值为max(m,n)+1
    public static int SolveLmsByGeneralizedInverse(double[] a, int m, int n, double[] b, double[] x,
                                                   double[] aa, double eps, double[] u, double[] v, int ka) {
        int i, j;
        i = GetGeneralizedInverseBySinglarityValueDecompositon(a, m, n, aa, eps, u, v, ka);
        if (i < 0) return (-1);
        for (i = 0; i <= n - 1; i++) {
            x[i] = 0.0;
            for (j = 0; j <= m - 1; j++)
                x[i] = x[i] + aa[i * m + j] * b[j];
        }
        return (1);
    }

    ///==========================================================================
    /// 8. 用分解法求解对称方程组
    /// 返回值小于0，工作失败；大于0，正常结束。
    /// a, n*n, 系数向量
    /// c, n*m, 右端常量（m组）。解完后存放方程组的m组解
    /// n, 方程组阶数
    /// m, 右端常量个数
    public static int SolveSymmetryMatrixByDecomposition(double[] a, int n, int m, double[] c) {
        int i, j, l, k, u, v, w, k1, k2, k3;
        double p;
        if (Math.abs(a[0]) + 1.0 == 1.0) {
            //printf("fail\n");
            return (-2);
        }
        for (i = 1; i <= n - 1; i++) {
            u = i * n;
            a[u] = a[u] / a[0];
        }
        for (i = 1; i <= n - 2; i++) {
            u = i * n + i;
            for (j = 1; j <= i; j++) {
                v = i * n + j - 1;
                l = (j - 1) * n + j - 1;
                a[u] = a[u] - a[v] * a[v] * a[l];
            }
            p = a[u];
            if (Math.abs(p) + 1.0 == 1.0) {
                //printf("fail\n");
                return (-2);
            }
            for (k = i + 1; k <= n - 1; k++) {
                u = k * n + i;
                for (j = 1; j <= i; j++) {
                    v = k * n + j - 1;
                    l = i * n + j - 1;
                    w = (j - 1) * n + j - 1;
                    a[u] = a[u] - a[v] * a[l] * a[w];
                }
                a[u] = a[u] / p;
            }
        }
        u = n * n - 1;
        for (j = 1; j <= n - 1; j++) {
            v = (n - 1) * n + j - 1;
            w = (j - 1) * n + j - 1;
            a[u] = a[u] - a[v] * a[v] * a[w];
        }
        p = a[u];
        if (Math.abs(p) + 1.0 == 1.0) {
            //printf("fail\n");
            return (-2);
        }
        for (j = 0; j <= m - 1; j++)
            for (i = 1; i <= n - 1; i++) {
                u = i * m + j;
                for (k = 1; k <= i; k++) {
                    v = i * n + k - 1;
                    w = (k - 1) * m + j;
                    c[u] = c[u] - a[v] * c[w];
                }
            }
        for (i = 1; i <= n - 1; i++) {
            u = (i - 1) * n + i - 1;
            for (j = i; j <= n - 1; j++) {
                v = (i - 1) * n + j;
                w = j * n + i - 1;
                a[v] = a[u] * a[w];
            }
        }
        for (j = 0; j <= m - 1; j++) {
            u = (n - 1) * m + j;
            c[u] = c[u] / p;
            for (k = 1; k <= n - 1; k++) {
                k1 = n - k;
                k3 = k1 - 1;
                u = k3 * m + j;
                for (k2 = k1; k2 <= n - 1; k2++) {
                    v = k3 * n + k2;
                    w = k2 * m + j;
                    c[u] = c[u] - a[v] * c[w];
                }
                c[u] = c[u] / a[k3 * n + k3];
            }
        }
        return (2);
    }


    ///==========================================================================
    /// 9. 用Levinson递推算法求解Toeplitz方程组，1.10
    /// 返回值小于0，工作失败；大于0，正常结束。
    /// t, 一维数组，长度为n，系数向量
    /// n, 方程组阶数
    /// b, 右端常量
    /// x, 返回方程组的解
    public static int SolveToeplitzEquationByLevinsonAlgorithm(double[] t, int n, double[] b, double[] x) {
        int i, j, k;
        double a, beta, q, c, h;
        double[] s = new double[n];
        double[] y = new double[n];
        a = t[0];
        if (Math.abs(a) + 1.0 == 1.0) {
            //free(s);
            //free(y);
            //printf("fail\n");
            return (-1);
        }
        y[0] = 1.0;
        x[0] = b[0] / a;
        for (k = 1; k <= n - 1; k++) {
            beta = 0.0;
            q = 0.0;
            for (j = 0; j <= k - 1; j++) {
                beta = beta + y[j] * t[j + 1];
                q = q + x[j] * t[k - j];
            }
            if (Math.abs(a) + 1.0 == 1.0) {
                //free(s);
                //free(y);
                //printf("fail\n");
                return (-1);
            }
            c = -beta / a;
            s[0] = c * y[k - 1];
            y[k] = y[k - 1];
            if (k != 1)
                for (i = 1; i <= k - 1; i++)
                    s[i] = y[i - 1] + c * y[k - i - 1];
            a = a + c * beta;
            if (Math.abs(a) + 1.0 == 1.0) {
                //free(s);
                //free(y);
                //printf("fail\n");
                return (-1);
            }
            h = (b[k] - q) / a;
            for (i = 0; i <= k - 1; i++) {
                x[i] = x[i] + h * s[i];
                y[i] = s[i];
            }
            x[k] = h * y[k];
        }
        //free(s);
        //free(y);
        return (1);
    }


    ///HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH
    /// 10. 求解线性最小二乘问题（用解对称方程组方法直接求解）
    /// less than 0,失败； greater than 0, 成功
    /// A, 一维向量，长度为 m*n
    /// m, 方程数目
    /// n, 参数数目
    /// b， 右端向量
    /// dampFactor,       阻尼系数
    /// weightedFactor,   加权系数
    /// Hx,  最小二乘解， Hx = Inverse(TA*A+d*I)*TA*b
    /// 其中，T为转置标置
    public static int HSolveLmsBySolvingSymmetrixMatrix(double[] A, int M, int N, double[] b,
                                                        double[] Hx, double weightedFactor, double dampFactor) {
        //Trace.assert(weightedFactor<=1 && weightedFactor>=0);
        assert weightedFactor <= 1 && weightedFactor >= 0;
        int i, j, k;
        double[] Hmatrix = new double[N * N];

        // 1. Initialize Hmatrix && Hx
        for (i = 0; i < N; i++) {
            for (j = 0; j < N; j++)
                Hmatrix[i * N + j] = 0.0f;
            Hx[i] = 0.0f;
        }
        double c1 = 1;
        for (k = 0; k < M; k++) {
            c1 *= weightedFactor;
            for (i = 0; i < N; i++) {
                for (j = i; j < N; j++)
                    Hmatrix[i * N + j] += (c1 * A[k * N + i] * A[k * N + j]);
                Hx[i] += (c1 * b[k] * A[k * N + i]);
            }
        }
        // get lower triangualr matrix from upper triangular matrix
        for (j = 1; j < N; j++)
            for (k = 0; k < j; k++)
                Hmatrix[j * N + k] = Hmatrix[k * N + j];
        // add dampFactor
        for (i = 0; i < N; i++)
            Hmatrix[i * N + i] += dampFactor;

        // 2. solve it
        int result = SolveSymmetryMatrixByDecomposition(Hmatrix, N, 1, Hx);

        //delete[] Hmatrix;
        return result;
    }


    ///==========================================================================
    /// 11. 用全选主元高斯消去法求解n阶线性代数方程组
    /// 返回值为0，表示原方程组的系数矩阵奇异；不为0，正常返回。
    /// a, n*n, 返回时将被破坏
    /// b, n，存放右端向量，返回方程组的解！
    public static int SolveEquationGroupByGaussMethod(double[] a, double[] b, int n) {
        int l, k, i, j, _is, p, q;
        double d, t;
        int[] js = new int[n];
        l = 1;
        _is = -1; // hhh
        for (k = 0; k <= n - 2; k++) {
            d = 0.0;
            for (i = k; i <= n - 1; i++)
                for (j = k; j <= n - 1; j++) {
                    t = Math.abs(a[i * n + j]);
                    if (t > d) {
                        d = t;
                        js[k] = j;
                        _is = i;
                    }
                }
            if (d + 1.0 == 1.0) l = 0;
            else {
                if (js[k] != k)
                    for (i = 0; i <= n - 1; i++) {
                        p = i * n + k;
                        q = i * n + js[k];
                        t = a[p];
                        a[p] = a[q];
                        a[q] = t;
                    }
                if (_is != k) {
                    for (j = k; j <= n - 1; j++) {
                        p = k * n + j;
                        q = _is * n + j;
                        t = a[p];
                        a[p] = a[q];
                        a[q] = t;
                    }
                    t = b[k];
                    b[k] = b[_is];
                    b[_is] = t;
                }
            }
            if (l == 0) {
                //free(js);
                //printf("fail\n");
                return (0);
            }
            d = a[k * n + k];
            for (j = k + 1; j <= n - 1; j++) {
                p = k * n + j;
                a[p] = a[p] / d;
            }
            b[k] = b[k] / d;
            for (i = k + 1; i <= n - 1; i++) {
                for (j = k + 1; j <= n - 1; j++) {
                    p = i * n + j;
                    a[p] = a[p] - a[i * n + k] * a[k * n + j];
                }
                b[i] = b[i] - a[i * n + k] * b[k];
            }
        }
        d = a[(n - 1) * n + n - 1];
        if (Math.abs(d) + 1.0 == 1.0) {
            //free(js);
            //printf("fail\n");
            return (0);
        }
        b[n - 1] = b[n - 1] / d;
        for (i = n - 2; i >= 0; i--) {
            t = 0.0;
            for (j = i + 1; j <= n - 1; j++)
                t = t + a[i * n + j] * b[j];
            b[i] = b[i] - t;
        }
        js[n - 1] = n - 1;
        for (k = n - 1; k >= 0; k--)
            if (js[k] != k) {
                t = b[k];
                b[k] = b[js[k]];
                b[js[k]] = t;
            }
        //free(js);
        return (1);
    }
}
