package zju.pso;

public class Particle {
    private double fitnessValue;
    private Velocity velocity;
    private Location location; // 位置就是粒子对应状态向量的值

    OptModel optModel;

    public Particle(OptModel optModel) {
        this.optModel = optModel;
    }

    public Velocity getVelocity() {
        return velocity;
    }

    public void setVelocity(Velocity velocity) {
        this.velocity = velocity;
    }

    public Location getLocation() {
        return location;
    }

    public void setLocation(Location location) {
        this.location = location;
    }

    // 根据当前位置计算适应度值
    public double getFitnessValue() {
        fitnessValue = optModel.evalObj(location);
        return fitnessValue;
    }
}
