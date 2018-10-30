package zju.topo;

import zju.dsmodel.DsConnectNode;

public class TopoNode {
    //
    private DsConnectNode cn;
    //名字
    private String name;
    //质量
    private double mass;
    //出度
    private int degree;

    private double xCoordinate;
    private double yCoordinate;

    private double xForce;
    private double yForce;

    public double getMass() {
        return mass;
    }

    public void setMass(double mass) {
        this.mass = mass;
    }

    public int getDegree() {
        return degree;
    }

    public void setDegree(int degree) {
        this.degree = degree;
    }

    public double getxCoordinate() {
        return xCoordinate;
    }

    public void setxCoordinate(double xCoordinate) {
        this.xCoordinate = xCoordinate;
    }

    public double getyCoordinate() {
        return yCoordinate;
    }

    public void setyCoordinate(double yCoordinate) {
        this.yCoordinate = yCoordinate;
    }

    public double getxForce() {
        return xForce;
    }

    public void setxForce(double xForce) {
        this.xForce = xForce;
    }

    public double getyForce() {
        return yForce;
    }

    public void setyForce(double yForce) {
        this.yForce = yForce;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public DsConnectNode getCn() {
        return cn;
    }

    public void setCn(DsConnectNode cn) {
        this.cn = cn;
    }
}
