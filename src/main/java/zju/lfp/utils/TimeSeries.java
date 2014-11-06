package zju.lfp.utils;

import static java.lang.System.arraycopy;


/**
 * Created by IntelliJ IDEA.
 * User: iceson
 * Date: 2005-12-12
 * Time: 9:54:41
 * To change this template use File | Settings | File Templates.
 */
public class TimeSeries {
    public double[] m_pSeries;
    public int m_nLen;


    public TimeSeries() {
        m_pSeries = null;
        m_nLen = 0;
    }

    public TimeSeries(TimeSeries series) {
        m_pSeries = null;
        m_nLen = 0;
        SetSeries(series.m_nLen);
        arraycopy(series.m_pSeries, 0, m_pSeries, 0, m_nLen);
    }

    public void SetSeries(int nLen) {
        assert(nLen > 0);
        if (m_nLen == nLen) return;
        m_nLen = nLen;
        m_pSeries = new double[m_nLen];
    }

    public void SetSeries(int nLen, double[] fSeries) {
        assert(nLen > 0 && fSeries != null);
        m_nLen = nLen;
        m_pSeries = fSeries;
    }

    public void SetSeries(int nLen, double[] fSeries, int nBeginPos) {
        assert(nLen > 0 && fSeries != null);
        m_nLen = nLen;
        m_pSeries = new double[m_nLen];
        System.arraycopy(fSeries, nBeginPos, m_pSeries, 0, m_nLen);
    }


    // 计算时间序列的期望 方差 偏度 峰度
    public double Mean() {
        assert (m_nLen > 0);
        double fMean = 0.0f;
        for (int i = 0; i < m_nLen; i++)
            fMean += m_pSeries[i];
        fMean /= m_nLen;
        return fMean;
    }

    public double Var(double fMean) {
        assert(m_nLen > 0);
        double fVar = 0.0f;
        for (int i = 0; i < m_nLen; i++)
            fVar += (m_pSeries[i] - fMean) * (m_pSeries[i] - fMean);
        fVar /= m_nLen;
        return fVar;
    }

    public double Moment3(double fMean, double fVar) {
        assert(m_nLen > 0);
        ///f1
        double f1 = 0.0f;
        for (int i = 0; i < m_nLen; i++) {
            double temp = m_pSeries[i] - fMean;
            f1 += temp * temp * temp;
        }
        f1 = f1 / (fVar * fVar * fVar * m_nLen);
        return f1;
    }

    public double Moment4(double fMean, double fVar) {
        assert(m_nLen > 0);
        ///f1
        double f1 = 0.0f;
        for (int i = 0; i < m_nLen; i++) {
            double temp = m_pSeries[i] - fMean;
            f1 += temp * temp * temp * temp;
        }
        f1 = f1 / (fVar * fVar * fVar * fVar * m_nLen);
        return f1;
    }

    //时间序列特性的检验  正态性 与 平稳性
    boolean IsNormal(double fError) {
        double m1, m2, m3, m4;
        m1 = Mean();
        m2 = Var(m1);
        m3 = Moment3(m1, m2);
        m4 = Moment4(m1, m2);
        return Math.abs(m3) <= fError && Math.abs(m4 - 3) <= fError;
    }

    boolean IsNormalWhiteNoise(double fError) {
        double m1, m2, m3, m4;
        m1 = Mean();
        m2 = Var(m1);
        m3 = Moment3(m1, m2);
        m4 = Moment4(m1, m2) - 3;
        return !(m3 - 2 >= fError || m4 - 2 >= fError);
    }


    public boolean IsWhiteNoise() {
        assert(m_nLen > 0);
        double[] acf = new double[m_nLen];
        CalAutoRelationFactor(acf, m_nLen);
        double temp = Math.sqrt(m_nLen);
        for (int i = 0; i < m_nLen; i++)
            acf[i] *= temp;
        int n1 = 0, n2 = 0;
        for (int i = 0; i < m_nLen; i++) {
            if (Math.abs(acf[i]) < 1) n1++;
            if (Math.abs(acf[i]) < 2) n2++;
        }
        if (1.0 * n1 / m_nLen < 0.6) return false;  /// 0.683
        if (1.0 * n2 / m_nLen < 0.9) return false;  /// 0.955
        return true;
    }

    //计算自相关
    public void CalAutoRelationFactor(double[] ACF, int len) {
        double s0 = 0.0f;
        for (int k = 0; k < m_nLen; k++)
            s0 += m_pSeries[k] * m_pSeries[k];
        ///
        ACF[0] = 1;
        for (int k = 1; k < len; k++) {
            double s = 0.0f;
            for (int t = 0; t < m_nLen - k; t++)
                s += m_pSeries[t] * m_pSeries[t + k];
            ACF[k] = s / s0;
        }
    }

    public void CalPartRelationFactor(double[] PACF, int len) {
        double[] ACF = new double[len + 1];
        try {
            CalRelationFactor(ACF, len + 1, PACF, len);
        }
        catch (Exception e) {
            System.out.println(e.toString());
        }
        //delete[] ACF;
    }

    public void CalRelationFactor(double[] ACF, int len1, double[] PACF, int len2) throws Exception {
        assert(len1 > len2);
        CalAutoRelationFactor(ACF, len1);
        ///
        double[] bb = new double[len2];
        System.arraycopy(ACF, 1, bb, 0, len2);
        if (MatrixOperation.SolveToeplitzEquationByLevinsonAlgorithm(ACF, len2, bb, PACF) < 0) 
            throw new Exception("SolvePartRelationFactor Failure!");
    }

    //标准化及恢复
    public void Normalize(double mean, double var) {
        assert(var > 0);
        double dev = Math.sqrt(var);
        for (int k = 0; k < m_nLen; k++)
            m_pSeries[k] = (m_pSeries[k] - mean) / dev;
    }

    public void N_Normalize(double mean, double var) {
        assert(var > 0);
        double dev = Math.sqrt(var);
        for (int k = 0; k < m_nLen; k++)
            m_pSeries[k] = m_pSeries[k] * dev + mean;
    }

    //差分及逆运算
    public void Different(int S, int D) {
        if (S == 0 || D == 0) return;  /// Need not to do differential
        for (int k = m_nLen - 1; k >= S; k--)
            m_pSeries[k] = m_pSeries[k] - m_pSeries[k - S];
        Different(S, D - 1);
    }

    public void Intergral(int S, int D) {
        if (S == 0 || D == 0) return;
        for (int k = S; k < m_nLen; k++)
            m_pSeries[k] = m_pSeries[k] + m_pSeries[k - S];
        Intergral(S, D - 1);
    }

    //子序列的求解
    public void GetSubSeries(int nBegin, int nGap, TimeSeries s, int nLen) {
        assert(nGap > 0);
        s.SetSeries(nLen);
        for (int i = 0; i < nLen; i++)
            s.m_pSeries[i] = 0;
        for (int n = 0; n < nLen; n++) {
            int k = nBegin + n * nGap;
            if (k >= m_nLen)
                break;
            s.m_pSeries[n] = m_pSeries[k];
        }
    }

    //序列的若干运算
    public double SquareSum() {
        double s = 0;
        for (int k = 0; k < m_nLen; k++)
            s += m_pSeries[k] * m_pSeries[k];
        return s;
    }

    public int MinElementPos() {
        int n = 0;
        for (int k = 1; k < m_nLen; k++)
            if (m_pSeries[k] < m_pSeries[n]) n = k;
        return n;
    }

    public int MaxElementPos() {
        int n = 0;
        for (int k = 1; k < m_nLen; k++)
            if (m_pSeries[k] > m_pSeries[n]) n = k;
        return n;
    }

    public double[] getData() {
        return m_pSeries;
    }
}
