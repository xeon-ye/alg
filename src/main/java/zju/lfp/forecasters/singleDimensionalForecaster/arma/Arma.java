package zju.lfp.forecasters.singleDimensionalForecaster.arma;

import zju.lfp.utils.MatrixOperation;
import zju.lfp.utils.TimeSeries;

public class Arma {

    private AForecastModel forecastModel;
    private int pout;
    private int qout;
    private int rout;

    public Arma() {
        forecastModel = AForecastModel._Model_Ar;
    }

    public void SetModel(AForecastModel m) {
        forecastModel = m;
    }

    public boolean Forecast(TimeSeries pData, TimeSeries pControl, TimeSeries pControlFore, TimeSeries pFore, int pMax) {
        double[] coeff = new double[3 * pMax - 1];
        double[] noise = new double[pData.m_nLen];
        // 1.
        double mean = pData.Mean();
        for (int k = 0; k < pData.m_nLen; k++)
            pData.m_pSeries[k] = (pData.m_pSeries[k] - mean);
        //pData.Normalize(mean,var);

        // 2.
        // 2.1 Calculate Model Parameter
//        int p = 0, q = 0, r = 0;
        for(int i=0;i<pData.m_nLen;i++) {
            if(Double.isNaN(pData.m_pSeries[i]))
            return false;
        }
        if (!BuildAll(pData, pControl, pMax, coeff, noise)) return false;
        // 2.2 forecast for steadible Series
        ForecastValue(pData, pControl, pControlFore, coeff, pout, qout, rout, noise, pFore);

        // 3.
        //pData.N_Normalize(mean,var);
        //pFore.N_Normalize(mean,var);
        for (int k = 0; k < pData.m_nLen; k++)
            pData.m_pSeries[k] = (pData.m_pSeries[k] + mean);
        for (int k = 0; k < pFore.m_nLen; k++)
            pFore.m_pSeries[k] = (pFore.m_pSeries[k] + mean);
        //
        return true;
    }

    boolean BuildAll(TimeSeries pData, TimeSeries pControl, int pMax, double[] coeff, double[] noise) {
        int len = pData.m_nLen;
        int p = -1;
        int q = 0;
        int r = 0;
        if (forecastModel == AForecastModel._Model_Ar2) {
            int minAchPoint = 0;
            double minAch = Math.pow(10, 9);//PInfiniteLarge;
            //
            //int myBase = Math.log(len);
            int myBase = 3;
            p = 0;
            int maxP = 31; //pMax;
            assert(maxP > p + myBase);
            while (maxP > p + myBase) {
                p += myBase;
                //
                //BuildArModelByAutoRelationFactor(pData,p,coeff);
                BuildArModelByYValue(pData, p, coeff);
                //
                double ach = GetSquareSum(pData, pControl, coeff, p, q, r, noise);
                ach = Math.log(ach) + (p + q + r + 1) * Math.log(len) / len;   // bic
                //ach = Math.log(ach) + (p+q+r+1)*2./len;      // aic
                //
                if (ach < minAch) {
                    minAch = ach;
                    minAchPoint = p / myBase;
                }
            }
            p = minAchPoint * myBase;
            //BuildArModelByAutoRelationFactor(pData,p,coeff);
            BuildArModelByYValue(pData, p, coeff);
        } else if (forecastModel == AForecastModel._Model_Ar) {
            int myBase = (int) Math.log(len);
            //int myBase = 1;
            if (myBase < 1) myBase = 1;
            p = 0;
            double originValue = Math.pow(10, 9);//PInfiniteLarge;
            assert(pMax > p + myBase);
            while (pMax > p + myBase) {
                p += myBase;
                BuildArModelByAutoRelationFactor(pData, p, coeff);
                //BuildArModelByYValue(pData,p,coeff);
                double ach = GetSquareSum(pData, pControl, coeff, p, q, r, noise);
                ach = Math.log(ach) + (p + q + r + 1) * Math.log(len) / len;   // bic
                //ach = Math.log(ach) + (p+q+r+1)*2./len;   // aic
                if (ach > originValue) {
                    p -= myBase;
                    BuildArModelByAutoRelationFactor(pData, p, coeff);
                    //BuildArModelByYValue(pData,p,coeff);
                    break;
                }
                originValue = ach;
            }
        } else if (forecastModel == AForecastModel._Model_LongAr) {
            int myBase = (int) Math.log(len);
            p = 0;
            assert(pMax > p + myBase);
            while (pMax > p + myBase) {
                p += myBase;
                BuildArModelByYValue(pData, p, coeff);
                GetNoise(pData, pControl, coeff, p, q, r, noise);
                TimeSeries t = new TimeSeries();
                t.SetSeries(len - p, noise, p);
                if (t.IsWhiteNoise()) break;
            }
            //
            double[] noise2 = new double[len];
            int PN = p;
            p = 2;
            q = 1;
            double originValue = Math.pow(10, 9);  //PInfiniteLarge;
            while (p < pMax) {
                BuildArmaModelByNoise(pData, noise2, PN, p, q, coeff);
                double ach = GetSquareSum(pData, pControl, coeff, p, q, r, noise);
                ach = len * Math.log(ach) + (p + q + r + 1) * Math.log(len);   // bic
                //ach = len*Math.log(ach) + (p+q+r+1)*2;   // aic
                if (ach > originValue) {
                    p -= 2;
                    q -= 2;
                    BuildArmaModelByNoise(pData, noise2, PN, p, q, coeff);
                    break;
                }
                originValue = ach;
                p += 2;
                q += 2;
            }
        } else if (forecastModel == AForecastModel._Model_Arma) {
            // 0.
            int nDataLen = pData.m_nLen;
            int nLen = pMax * 2;
            if (nLen >= nDataLen) nLen = nDataLen - 2;
            double[] b = new double[nLen];
            double[] coeffBackup = new double[pMax * 2];
            // 1
            BuildArModelByYValue(pData, nLen, b);
            //
            double originValue = Math.pow(10, 9);//PInfiniteLarge;
            double tValue = 0.0;
            int rankStep = 2;
            p = 2;
            q = 1;
            while (p < pMax) {
                if (!BuildArmaModel(pData, b, nLen, p, q, coeff)) return false;
                //
                tValue = NLmsObjectiveValue(pData, p, q, nLen, p + q, coeff);
                if (FTest(originValue, tValue, p + q + 1, 2 * rankStep, nDataLen)) {
                    p -= rankStep;
                    q -= rankStep;
                    for (int i = 0; i < p + q + 1; i++)
                        coeff[i] = coeffBackup[i];
                    break;
                }
                originValue = tValue;
// arraycopy为系统推荐方法   罗军 2005 12 14
                System.arraycopy(coeff, 0, coeffBackup, 0, p + q + 1);
                p += rankStep;
                q += rankStep;
            }
        }
        assert(p != -1);
        pout = p;
        qout = q;
        rout = r;
        //
        return true;
    }

    void ForecastValue(TimeSeries pData, TimeSeries pControl, TimeSeries pControlFore,
                       double[] fCoeff, int p, int q, int r, double[] noise, TimeSeries pFore) {
        int len;
        if (q == 0) {
            TimeSeries data = pData;
            for (int k = 0; k < pFore.m_nLen; k++) {
                double dValue = 0;
                len = data.m_nLen;
                //
                for (int j = 0; j < p; j++) {
                    int lIndex = len + k - (j + 1);
                    if (lIndex >= 0 && lIndex < len) dValue += fCoeff[j] * data.m_pSeries[lIndex];
                    else if (lIndex >= len) dValue += fCoeff[j] * pFore.m_pSeries[lIndex - len];
                }
                //
                if (r > 0) {
                    //coeff = &fCoeff[p+q];
                    int myBase = p + q;
                    for (int j = 0; j < r; j++) {
                        int lIndex = len + k - (j + 1);
                        if (lIndex >= 0 && lIndex < len) dValue += fCoeff[myBase + j] * pControl.m_pSeries[lIndex];
                        else if (lIndex >= len) dValue += fCoeff[myBase + j] * pControlFore.m_pSeries[lIndex - len];
                    }
                }
                pFore.m_pSeries[k] = dValue;
            }
            //
            return;
        }

        // 1. get G
        int dataLen = pData.m_nLen;
        len = dataLen + pFore.m_nLen;
        double[] G = new double[len];
        G[0] = 1;
        for (int l = 1; l < len; l++) {
            G[l] = ((l <= q) ? -1 * fCoeff[p + l - 1] : 0);
            for (int j = 1; j <= l; j++)
                G[l] += G[l - j] * (j <= p ? fCoeff[j - 1] : 0);
        }
        // 2.
        GetNoise(pData, pControl, fCoeff, p, q, r, noise);
        // 3. get pFore
        for (int l = 0; l < pFore.m_nLen; l++) {
            double dValue = 0;
            for (int k = 0; k < dataLen; k++)
                dValue += G[l + k] * noise[dataLen - 1 - k];
            pFore.m_pSeries[l] = dValue;
        }
    }

    double GetNoise(TimeSeries pData, TimeSeries pControl, double[] fCoeff, int p, int q, int r, double[] pNoise) {
        TimeSeries data;
        data = pData;
        for (int k = 0; k < pData.m_nLen; k++)
            pNoise[k] = 0;
        double sValue = 0;
        double LargeError = 1000000;
        //
        int nBegin = 0;
        int nEnd = pData.m_nLen;
        for (int k = nBegin; k < nEnd; k++) {
            double dValue = pData.m_pSeries[k];
            //
            for (int j = 0; j < p; j++) {
                int lIndex = k - (j + 1);
                if (lIndex >= nBegin) dValue -= fCoeff[j] * data.m_pSeries[lIndex];
                if (dValue > LargeError) dValue = LargeError;
                if (dValue < -1 * LargeError) dValue = -1 * LargeError;
            }
            //
            if (q > 0) {
                //coeff = &fCoeff[p];
                int myBase = p;
                for (int j = 0; j < q; j++) {
                    int lIndex = k - (j + 1);
                    if (lIndex >= nBegin) dValue += fCoeff[myBase + j] * pNoise[lIndex];
                    if (dValue > LargeError) dValue = LargeError;
                    if (dValue < -1 * LargeError) dValue = -1 * LargeError;
                }
            }
            //
            if (r > 0) {
                //coeff = &fCoeff[p+q];
                int myBase = p + q;
                for (int j = 0; j < r; j++) {
                    int lIndex = k - (j + 1);
                    if (lIndex >= nBegin) dValue += fCoeff[myBase + j] * pControl.m_pSeries[lIndex];
                    if (dValue > LargeError) dValue = LargeError;
                    if (dValue < -1 * LargeError) dValue = -1 * LargeError;
                }
            }
            pNoise[k] = dValue;
            sValue += dValue * dValue;
        }
        return sValue;
    }

    double GetSquareSum(TimeSeries pData, TimeSeries pControl, double[] fCoeff, int p, int q, int r, double[] noise) {
        GetNoise(pData, pControl, fCoeff, p, q, r, noise);
        //
        int len = pData.m_nLen;
        int nDiscard = 10 * q;
        TimeSeries t = new TimeSeries();
        t.SetSeries(len - nDiscard, noise, nDiscard);
        double ach = t.SquareSum() / (len - nDiscard);
        return ach;
    }

    boolean BuildArModelByYValue(TimeSeries pData, int p, double[] arCoeff) {
        // 0.
        boolean bSuccess = true;
        //int equationNum = p*2;
        //if ( equationNum > pData.m_nLen-1-p ) equationNum = pData.m_nLen-1-p;
        int equationNum = pData.m_nLen - 1 - p;
        int paraNum = p;
        assert(equationNum > p);

        // 1.
        double[] a = new double[equationNum * paraNum];
        double[] b = new double[equationNum];
        double[] x = new double[paraNum];

        // 2.1 get b,a
        for (int i = 0; i < equationNum; i++) {
            int hhh = pData.m_nLen - 1 - i;
            b[i] = pData.m_pSeries[hhh];
            for (int j = 0; j < p; j++)
                a[i * paraNum + j] = pData.m_pSeries[hhh - 1 - j];
        }
        // 2.2 solve it
        if (MatrixOperation.HSolveLmsBySolvingSymmetrixMatrix(a, equationNum, paraNum, b, x, 1, 0) == 0)
            bSuccess = false;
        // 2.3 get result
        for (int i = 0; i < paraNum; i++)
            arCoeff[i] = x[i];

        // 3.
        //delete[] a;
        //delete[] b;
        //delete[] x;

        return bSuccess;
    }

    boolean BuildArModelByAutoRelationFactor(TimeSeries pData, int p, double[] arCoeff) {
        boolean bSuccess = true;
        int acfNum = (int) (p * 1.3 + 3);
        double[] acf = new double[acfNum];
        pData.CalAutoRelationFactor(acf, acfNum);

        // 0.
        int equationNum = acfNum - 1;
        int paraNum = p;
        assert(equationNum > paraNum);

        // 1.
        double[] a = new double[equationNum * paraNum];
        double[] b = new double[equationNum + 1];
        double[] Q = new double[equationNum * equationNum];

        // 2.0 get b
        for (int i = 0; i < equationNum + 1; i++)
            b[i] = acf[i];
        // 2.1 get a
        for (int i = 0; i < equationNum; i++)
            for (int j = 0; j < paraNum; j++) {
                int hhh = j - i;
                if (hhh < 0) hhh *= -1;
                a[i * paraNum + j] = b[hhh];
            }
        // 2.2 solve it
        double[] bb = new double[equationNum];
        for (int i = 0; i < equationNum; i++)
            bb[i] = b[1 + i];
        if (MatrixOperation.SolveLmsByQR(a, equationNum, paraNum, bb, Q) == 0)
            bSuccess = false;
        // 2.3 get result
        for (int i = 0; i < paraNum; i++)
            arCoeff[i] = bb[i];

        // 3.
        //delete[] acf;
        //delete[] a;
        //delete[] b;
        //delete[] Q;

        return bSuccess;
    }


    boolean BuildArxModelByYValue(TimeSeries pData, TimeSeries pControl, int p, int r, double[] coeff) {
        boolean bSuccess = true;

        // 0.
        int n = p + r;
        int m = (int) (n * 1.8);  // n*5;
        assert(m > n);
        if (m > pData.m_nLen - r - p) {
            m = pData.m_nLen - r - p;
        }

        // 1.
        double[] a = new double[m * n];
        double[] b = new double[pData.m_nLen];
        double[] Q = new double[m * m];

        // 2.1 get b,a
        for (int i = 0; i < m; i++) {
            int hhh = pData.m_nLen - 1 - i;
            //
            b[i] = pData.m_pSeries[hhh];
            for (int j = 0; j < n; j++) {
                if (j < p) a[i * n + j] = pData.m_pSeries[hhh - 1 - j];
                else
                    a[i * n + j] = pControl.m_pSeries[hhh - 1 - (j - p)];
            }
        }
        // 2.2 solve it
        if (MatrixOperation.SolveLmsByQR(a, m, n, b, Q) == 0) bSuccess = false;
        // 2.3 get result
        //for( int i=0;i<p;i++) coeff[i] = b[i];
        for (int i = 0; i < n; i++)
            coeff[i] = b[i];

        // 3.
        //delete[] a;
        //delete[] b;
        //delete[] Q;
        return bSuccess;
    }

    boolean BuildArxModelByYValue2(TimeSeries pData, TimeSeries pControl, int p, int r, double[] coeff) {
        boolean bSuccess = true;

        // 0.
        int rRank = 2;
        int n = p + r * rRank;
        int m = n * 12;  // n*1.8;
        assert(m > n);
        if (m > pData.m_nLen - r - p) {
            m = pData.m_nLen - r - p;
        }

        // 1.
        double[] a = new double[m * n];
        double[] b = new double[pData.m_nLen];
        double[] x = new double[n];

        // 2.1 get b,a
        for (int i = 0; i < m; i++) {
            int hhh = pData.m_nLen - 1 - i;
            //
            b[i] = pData.m_pSeries[hhh];
            for (int j = 0; j < p + r; j++) {
                if (j < p) a[i * n + j] = pData.m_pSeries[hhh - 1 - j];
                else {
                    double dTemp = 1;
                    for (int k = 0; k < rRank; k++) {
                        dTemp *= pControl.m_pSeries[hhh - 1 - (j - p)];
                        a[i * n + (p + (j - p) * rRank + k)] = dTemp;
                    }
                }
            }
        }
        // 2.2 solve it
        if (MatrixOperation.HSolveLmsBySolvingSymmetrixMatrix(a, m, n, b, x, 1, 0) == 0)
            bSuccess = false;
        // 2.3 get result
        for (int i = 0; i < n; i++)
            coeff[i] = x[i];
        return bSuccess;
    }

    boolean BuildArxModelByRelationFactor(TimeSeries pData, TimeSeries pControl, int p, int r, double[] coeff) {
        boolean bSuccess = true;

        // 0.
        assert(pData.m_nLen == pControl.m_nLen);
        int len = pData.m_nLen;
        int n = p + r;
        int m = (int) (n * 1.9);  // n*5;
        assert(m > n);
        if (m > len - r - p) {
            m = len - r - p;
        }

        // 1.
        double[] a = new double[m * n];
        double[] b = new double[len];
        double[] Q = new double[m * m];

        //
        pData.CalAutoRelationFactor(b, len);
        // E[u(t)*y(t-k)], k=0,...,len
        double s0 = 0.0f;
        for (int k = 0; k < len; k++)
            s0 += pData.m_pSeries[k] * pData.m_pSeries[k];
        double[] b21 = new double[pControl.m_nLen];
        double[] b22 = new double[pControl.m_nLen];
        for (int k = 0; k < len; k++) {
            double s = 0.0f;
            for (int t = 0; t < len - k; t++) // y(t)*u(t-k)
                s += pControl.m_pSeries[t] * pData.m_pSeries[t + k];
            b21[k] = s / s0;
        }
        for (int k = 0; k < len; k++) {
            double s = 0.0f;
            for (int t = 0; t < len - k; t++) // u(t)*y(t-k)
                s += pData.m_pSeries[t] * pControl.m_pSeries[t + k];
            b22[k] = s / s0;
        }
        // 2.1 get b,a
        for (int i = 0; i < m; i++) {
            //int hhh_pic = len-1-i;
            //b[i] = pData.m_pSeries[hhh_pic];
            for (int j = 0; j < n; j++) {
                int gap = j - i;
                if (gap < 0) gap *= -1;
                if (j < p) a[i * n + j] = b[gap];//pData.m_pSeries[hhh_pic-1-j];
                else {
                    gap = j - p - i;
                    if (gap > 0) a[i * n + j] = b22[gap];        // lower corner , y rank more than u
                    else
                        a[i * n + j] = b21[-1 * gap];     // upper corner,  y rank less than u
                }
            }
        }
        //delete[] b21;
        //delete[] b22;
        // 2.2 solve it
        double[] bb = new double[m];
        for (int i = 0; i < m; i++)
            bb[i] = b[1 + i];
        if (MatrixOperation.SolveLmsByQR(a, m, n, bb, Q) == 0) bSuccess = false;
        // 2.3 get result
        //for( int i=0;i<p;i++) coeff[i] = b[i+1];
        for (int i = 0; i < n; i++)
            coeff[i] = bb[i];
        return bSuccess;
    }

    boolean BuildArmaModelByNoise(TimeSeries pData, double[] noise, int PN,
                                  int p, int q, double[] coeff) {
        boolean bSuccess = true;

        // 1.
        int len = pData.m_nLen;
        int m = p + q;
        int n = p + q;
        double[] a = new double[m * n];
        double[] b = new double[m];

        // 2.1 get a,b
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                a[i * n + j] = 0;
        for (int i = 0; i < m; i++)
            b[i] = 0;
        //
        for (int j = 0; j < p; j++) {
            for (int t = PN + q - 1; t < len; t++)
                b[j] += pData.m_pSeries[t] * pData.m_pSeries[t - j];
            for (int h = 0; h < p; h++) {
                for (int t = PN + q - 1; t < len; t++)
                    a[j * n + h] += pData.m_pSeries[t - 1 - h] * pData.m_pSeries[t - j];
            }
            for (int h = 0; h < q; h++) {
                for (int t = PN + q - 1; t < len; t++)
                    a[j * n + p + h] += -1 * noise[t - 1 - h] * pData.m_pSeries[t - j];
            }
        }
        for (int j = 0; j < q; j++) {
            for (int t = PN + q - 1; t < len; t++)
                b[p + j] += pData.m_pSeries[t] * noise[t - j];
            for (int h = 0; h < p; h++) {
                for (int t = PN + q - 1; t < len; t++)
                    a[(p + j) * n + h] += pData.m_pSeries[t - 1 - h] * noise[t - j];
            }
            for (int h = 0; h < q; h++) {
                for (int t = PN + q - 1; t < len; t++)
                    a[(p + j) * n + p + h] += -1 * noise[t - 1 - h] * noise[t - j];
            }
        }
        // 2.2 solve it
        assert(m == n);
        if (MatrixOperation.SolveEquationGroupByGaussMethod(a, b, n) == 0) bSuccess = false;
        // 2.3 get result
        for (int i = 0; i < n; i++)
            coeff[i] = b[i];

        // 3. keep inverse
        if (q > 0) {
            for (int i = 0; i < q; i++)
                a[i] = -1 * coeff[p + (q - 1 - i)];
            a[q] = 1;
            AdjustEquationFactorToGuaranteeItsRootsInCircle(q, a);
            for (int i = 0; i < q; i++)
                coeff[p + i] = -1 * a[q - 1 - i];
        }

        //delete[] a;
        //delete[] b;
        //
        return bSuccess;
    }

    boolean BuildArmaModel(TimeSeries pData, double[] arCoeff, int nLen, int p, int q, double[] coeff) {
        boolean bSuccess = true;
        // 0.
        int m;
        m = nLen;
        int n = p + q;
        assert(m > n);
        double[] a = new double[m * n];
        double[] b = new double[m];
        double[] Q = new double[m * m];

        // 1.
        for (int i = 0; i < m; i++)
            b[i] = arCoeff[i];

        // 2. using inverse-function method, get this model parameter
        // 2.1 get a
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                a[i * n + j] = 0.0;
        //
        for (int j = 0; j < n; j++) {
            // the first p column
            if (j < p) {
                a[j * n + j] = 1;
                continue;
            }
            // if j >= p
            int k = j - p;
            a[k * n + j] = -1;
            int hhh = 0;
            for (int i = k + 1; i < m; i++)
                a[i * n + j] = b[hhh++];
        }
        // 2.2 solve it
        if (MatrixOperation.SolveLmsByQR(a, m, n, b, Q) == 0) bSuccess = false;
        // 2.3 get result
        for (int i = 0; i < n; i++)
            coeff[i] = b[i];

        // 3. keep inverse
        if (q > 0) {
            for (int i = 0; i < q; i++)
                a[i] = -1 * coeff[p + (q - 1 - i)];
            a[q] = 1;
            AdjustEquationFactorToGuaranteeItsRootsInCircle(q, a);
            for (int i = 0; i < q; i++)
                coeff[p + i] = -1 * a[q - 1 - i];
        }
        coeff[p + q] = 0;
        if (SolveNonlinearLmsByGaussNewtonMethodWithDampFactor(pData, p, q, pData.m_nLen - p, p + q + 1, coeff, Math.pow(10, -4)) <= 0)
            System.out.println("非线性最小二乘法未能正确完成！");
        return bSuccess;
    }

    int SolveNonlinearLmsByGaussNewtonMethodWithDampFactor(TimeSeries pData, int _p, int _q, int equationNum, int paraNum, double[] para, double eps) {
        int nSuccess = 1;
        double s0 = 0, s1 = 0;
        double[] sita0 = new double[paraNum];
        double[] sita1 = new double[paraNum];
        double[] deltaSita = new double[paraNum];
        double[] Jacobi = new double[equationNum * paraNum];
        double[] noise = new double[equationNum];
        //
        for (int i = 0; i < paraNum; i++)
            sita0[i] = para[i];
        //
        int MaxCalNumWhileEnlargingDampFactor = 10;
        int calNumWhileEnlargingDampFactor = 0;
        double dampFactor = 10;
        boolean bNewInitialValue = true;
        while (true) {
            if (bNewInitialValue) {
                NLmsVariableJacobiMatrix(pData, _p, _q, equationNum, paraNum, sita0, Jacobi);
                NLmsEquationResidualError(pData, _p, _q, equationNum, paraNum, sita0, noise);
                s0 = NLmsObjectiveValue(pData, _p, _q, equationNum, paraNum, sita0);
            }
            if (MatrixOperation.HSolveLmsBySolvingSymmetrixMatrix(Jacobi, equationNum, paraNum, noise, deltaSita, 1, dampFactor) < 0)
            {
                nSuccess = -1;
                break;
            }
            for (int i = 0; i < paraNum; i++)
                sita1[i] = sita0[i] + deltaSita[i];
            s1 = NLmsObjectiveValue(pData, _p, _q, equationNum, paraNum, sita1);
            if (Math.abs(s1 - s0) < eps) break;
            if (s1 < s0) {
                dampFactor *= 0.5;
                for (int i = 0; i < paraNum; i++)
                    sita0[i] = sita1[i];
                bNewInitialValue = true;
            } else  // s1 >= s0
            {
                if (bNewInitialValue) calNumWhileEnlargingDampFactor = 0;
                bNewInitialValue = false;
                calNumWhileEnlargingDampFactor++;
                if (calNumWhileEnlargingDampFactor < MaxCalNumWhileEnlargingDampFactor)
                    dampFactor *= 1.5;
                else {
                    // cal detaSita * JacobiMatrix * noise
                    double f1 = 0;
                    for (int p = 0; p < paraNum; p++) {
                        double fValue = 0;
                        for (int i = 0; i < equationNum; i++)
                            fValue += Jacobi[i * paraNum + p] * noise[i];
                        f1 += deltaSita[p] * fValue;
                    }
                    double lambda = f1 / (s1 - s0 + 2 * f1);
                    for (int i = 0; i < paraNum; i++)
                        sita1[i] = sita0[i] + lambda * deltaSita[i];
                    s1 = NLmsObjectiveValue(pData, _p, _q, equationNum, paraNum, sita1);
                    if (s1 >= s0) {
                        //for (int k = 0; k < 13; k++) {
                        for (int k = 0; k < 7; k++) {
                            lambda /= 2;
                            for (int i = 0; i < paraNum; i++)
                                sita1[i] = sita0[i] + lambda * deltaSita[i];
                            s1 = NLmsObjectiveValue(pData, _p, _q, equationNum, paraNum, sita1);
                            if (s1 < s0) break;
                        }
                        if (s1 >= s0) {
                            for (int i = 0; i < paraNum; i++)
                                sita1[i] = sita0[i];
                            break;
                        }
                    }
                    dampFactor *= 0.5;
                    for (int i = 0; i < paraNum; i++)
                        sita0[i] = sita1[i];
                    bNewInitialValue = true;
                }
            } // s1 >= s0
        }

        for (int i = 0; i < paraNum; i++)
            para[i] = sita1[i];
        //
        //delete[] sita0;
        //delete[] sita1;
        //delete[] deltaSita;
        //delete[] Jacobi;
        //delete[] noise;
        return nSuccess;
    }

    double NLmsObjectiveValue(TimeSeries pData, int p, int q, int equationNum, int paraNum, double[] coeff) {
        double[] d = new double[equationNum];
        NLmsEquationResidualError(pData, p, q, equationNum, paraNum, coeff, d);
        double fValue = 0;
        for (int i = 0; i < equationNum; i++)
            fValue += d[i] * d[i];
        //delete[] d;
        return fValue;
        //
        /*
              int nEnd = pData.m_nLen;
              int nBegin = nEnd - equationNum;
              assert(nBegin >= p);
              double[] u = new double[q];
              for ( int i=0;i<q;i++) u[i] = 0;
              double Sk = 0,Uk,Yk;
              for ( int k=nBegin;k<nEnd;k++)
              {
                  // 1.
                  Yk = pData.m_pSeries[k]-coeff[p+q];
                  for (int i=0;i<p;i++) Yk -= coeff[i]*(pData.m_pSeries[k-1-i]-coeff[p+q]);
                  // 2.
                  Uk = Yk;
                  for ( int i=0;i<q;i++) Uk += coeff[p+i]*u[q-1-i];  // sita1*u[k-1]
                  for ( int i=0;i<q-1;i++) u[i] = u[i+1];
                  u[q-1] = Uk;
                  // 3.
                  Sk += Uk*Uk;
                  if ( Sk > LargeValue ) break;
              }
              //delete[] u;
              return Sk;
          */
    }

    void NLmsEquationResidualError(TimeSeries pData, int p, int q, int m, int n, double[] coeff, double[] d) {
        double[] noise = new double[pData.m_nLen];
        GetNoise(pData, null, coeff, p, q, 0, noise);
        for (int i = 0; i < m; i++)
            d[i] = noise[pData.m_nLen - m + i];
    }

    void NLmsVariableJacobiMatrix(TimeSeries pData, int p, int q, int m, int n, double[] coeff, double[] pM) {
        double LargeError = Math.pow(10, 6);
        double[] noise = new double[pData.m_nLen];
        GetNoise(pData, null, coeff, p, q, 0, noise);
        int len = pData.m_nLen;
        for (int row = 0; row < m; row++) {
            int t = len - m + row;
            // ×????é?????ó??
            for (int j = 0; j < p; j++) {
                double dValue = 0;
                if (t - (j + 1) >= 0) dValue += pData.m_pSeries[t - (j + 1)];     // specially
                for (int i = 0; i < q; i++) {
                    int prevRow = row - (i + 1);
                    if (prevRow >= 0) dValue += coeff[p + i] * pM[prevRow * n + j];
                    if (dValue > LargeError) dValue = LargeError;
                    if (dValue < -1 * LargeError) dValue = -1 * LargeError;
                }
                pM[row * n + j] = dValue;
            }
            // MA?????ó??
            for (int i = 0; i < q; i++) {
                double dValue = 0;
                if (t - (i + 1) >= 0) dValue -= noise[t - (i + 1)];  // specially
                for (int j = 0; j < q; j++) {
                    int prevRow = row - (j + 1);
                    if (prevRow >= 0) dValue += coeff[p + j] * pM[prevRow * n + p + i];
                    if (dValue > LargeError) dValue = LargeError;
                    if (dValue < -1 * LargeError) dValue = -1 * LargeError;
                }
                pM[row * n + (p + i)] = dValue;
            }
            // ???ù???ó??
            double dValue2 = 1;
            for (int i = 0; i < p; i++)
                dValue2 -= coeff[i];
            for (int j = 0; j < q; j++) {
                int prevRow = row - (j + 1);
                if (prevRow >= 0) dValue2 += coeff[p + j] * pM[prevRow * n + p + q];
                if (dValue2 > LargeError) dValue2 = LargeError;
                if (dValue2 < -1 * LargeError) dValue2 = -1 * LargeError;
            }
            pM[row * n + (p + q)] = dValue2;
        }
        //delete[] noise;
    }

    void AdjustEquationFactorToGuaranteeItsRootsInCircle(int equationRank, double[] equationFactor) {
        if (equationRank <= 0) return;
        //
        int equationFactorLen = equationRank + 1;
        double[]  b = new double[equationFactorLen];
        double[] xu = new double[equationRank];
        double[] xv = new double[equationRank];
        double eps = Math.pow(10, -4);
        int jt;
        jt = 300;
        boolean bAdjusted = false;
        eps = Math.pow(10, -6);
        double[] a = equationFactor;
        int q;
        q = equationRank;
        for (int i = 0; i < equationRank; i++) {
            double r = xu[i] * xu[i] + xv[i] * xv[i];
            if (r <= 1) continue;
            //
            bAdjusted = true;
            if (Math.abs(xv[i]) < eps) // one real root
            {
                b[0] = 1;
                for (int k = 1; k < q; k++)
                    b[k] = xu[i] * b[k - 1] + a[k];
                b[q] = 0;
                for (int k = 1; k <= q; k++)
                    a[k] = -1 / xu[i] * b[k - 1] + b[k];
            } else {
                boolean bDealed = false;
                for (int j = 0; j < i; j++) {
                    if (Math.abs(xu[j]-xu[i]) < eps && Math.abs(xv[j] + xv[i]) < eps) {
                        bDealed = true;
                        break;
                    }
                }
                if (bDealed) continue;
                //
                double c1 = 2 * xu[i], c2 = r;
                b[0] = 1;
                b[1] = a[1] + c1 * b[0] - 0;
                for (int j = 2; j < q - 1; j++)
                    b[j] = a[j] + c1 * b[j - 1] - c2 * b[j - 2];
                b[q - 1] = b[q] = 0;
                a[1] = b[1] - c1 / c2 * b[0] + 0;
                for (int j = 2; j <= q; j++)
                    a[j] = b[j] - c1 / c2 * b[j - 1] + 1.0 / c2 * b[j - 2];
            }
        }
    }

    boolean FTest(double asValue, double a, int r, int s, int N) {
        final int M = 6;
        double[][] f = {
                {4.96, 4.35, 4.17, 4.08, 4.00, 3.92, 3.84},
                {4.10, 3.49, 3.32, 3.23, 3.15, 3.07, 3.00},
                {3.71, 3.10, 2.92, 2.84, 2.76, 2.68, 2.60},
                {3.48, 2.87, 2.69, 2.61, 2.53, 2.45, 2.37}
        };
        int[] mValue = {10, 20, 30, 40, 60, 120};
        double F = (asValue - a) / s / (a / (N - r));

        // get FNormal
        assert(s - 1 < 4);
        double FNormal = f[s - 1][M];
        for (int i = 0; i < M; i++)
            if (N - r < mValue[i]) FNormal = f[s - 1][i];

        // compare
        if (F <= FNormal) return true;
        return false;
    }
}
