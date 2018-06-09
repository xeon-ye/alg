package zju.pso;

/**
 * @Description: 粒子类
 * @Author: Fang Rui
 * @Date: 2018/6/7
 * @Time: 17:43
 */

public class Particle {
    private Velocity velocity;
    private Location location; // 位置就是粒子对应状态向量的值


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

}
