package zju.dsntp; /**
 * Created by meditation on 2017/12/18.
 */

import java.util.Date;

/**
 * 　　* 本线程设置了一个超时时间
 * 　　* 该线程开始运行后，经过指定超时时间，
 * 　　* 该线程会抛出一个未检查异常通知调用该线程的程序超时
 * 　　* 在超时结束前可以调用该类的cancel方法取消计时
 * 　　* @author solonote
 */
public class TimekeeperThread extends Thread {
    //计时器超时时间
    private long timeInterval;
    //
    private long lastTime;
    //计时是否被取消
    private boolean isStopped = false;
    //当计时器超时时抛出的异常
    private TimeoutException timeoutException;

    //
    ParticleInTSC particleInTSC;

    /**
     * 　　* 构造器
     * 　　* @param timeout 指定超时的时间
     */
    public TimekeeperThread(long timeInterval, TimeoutException timeoutErr, ParticleInTSC particle) {
        super();
        this.timeInterval = timeInterval;
        this.timeoutException = timeoutErr;
        this.particleInTSC = particle;
        //设置本线程为守护线程
        this.setDaemon(true);
    }

    /**
     * 取消计时
     */
    public synchronized void updateTime() {
        lastTime = System.currentTimeMillis();
    }

    /**
     * 启动超时计时器
     */
    public void run() {
        System.out.println(Thread.currentThread().getName() + " 守护睡眠 " + new Date());

        try {
            Thread.sleep(timeInterval);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
        if (System.currentTimeMillis() - lastTime > timeInterval) {
            //particleInTSC.isTooLong = true;
        }

    }
}
