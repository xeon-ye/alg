package zju.tmodel;

import zju.bpamodel.swi.Generator;

/**
 * Created by IntelliJ IDEA.
 * User: Dong Shufeng
 * Date: 12-7-16
 */
public class TGenModel {
    private Generator generator;

    public TGenModel() {
    }

    public TGenModel(Generator generator) {
        this.generator = generator;
    }

    double ud;
    double uq;
    double id;
    double iq;
    double eq;
    double eqd;
    double delta;

    double eqp;
    double omega;

    /**
     * This method assume ra is equal to zero.
     * @param uAmpl voltage amplitude of generator bus
     * @param uAngle voltage angle in arc of generator bus
     * @param iAmpl current amplitude
     * @param iAngle current angle in arc
     */
    public void cal(double uAmpl, double uAngle, double iAmpl, double iAngle, double baseMva) {
        double fai = uAngle - iAngle;
        double xq = generator.getBaseMva() > 0.0 ? generator.getXq() * baseMva / generator.getBaseMva() : generator.getXq();
        double xd = generator.getBaseMva() > 0.0 ? generator.getXd() * baseMva / generator.getBaseMva() : generator.getXd();
        double eqdx = uAmpl * Math.cos(uAngle) - xq * iAmpl * Math.sin(iAngle);
        double eqdy = uAmpl * Math.sin(uAngle) + xq * iAmpl * Math.cos(iAngle);
        delta = Math.atan(eqdy / eqdx);
        eqd = Math.sqrt(eqdx * eqdx + eqdy * eqdy);
        double angleUAndEq = delta - uAngle;
        uq = uAmpl * Math.cos(angleUAndEq);
        ud = uAmpl * Math.sin(angleUAndEq);
        id = iAmpl * Math.sin(fai + angleUAndEq);
        iq = iAmpl * Math.cos(fai + angleUAndEq);
        eq = uq + xd * id;
    }

    public void calThirdOrderInitial() {
        eqp = uq + generator.getXdp() * id + generator.getRa() * iq;
        omega = 1.0;
    }

    public Generator getGenerator() {
        return generator;
    }

    public void setGenerator(Generator generator) {
        this.generator = generator;
    }

    public double getDelta() {
        return delta;
    }

    public void setDelta(double delta) {
        this.delta = delta;
    }

    public double getUd() {
        return ud;
    }

    public void setUd(double ud) {
        this.ud = ud;
    }

    public double getUq() {
        return uq;
    }

    public void setUq(double uq) {
        this.uq = uq;
    }

    public double getId() {
        return id;
    }

    public void setId(double id) {
        this.id = id;
    }

    public double getIq() {
        return iq;
    }

    public void setIq(double iq) {
        this.iq = iq;
    }

    public double getEq() {
        return eq;
    }

    public void setEq(double eq) {
        this.eq = eq;
    }

    public double getEqd() {
        return eqd;
    }

    public void setEqd(double eqd) {
        this.eqd = eqd;
    }

    public double getEqp() {
        return eqp;
    }

    public void setEqp(double eqp) {
        this.eqp = eqp;
    }

    public double getOmega() {
        return omega;
    }

    public void setOmega(double omega) {
        this.omega = omega;
    }
}
