package zju.forecast;

import java.util.Date;

public class Weather {
    private Date dtime;
    private double windSpeed;
    private double temperature;
    private double solarIrradiance;
    private String weather;

    public Date getDtime() {
        return dtime;
    }

    public void setDtime(Date dtime) {
        this.dtime = dtime;
    }

    public double getWindSpeed() {
        return windSpeed;
    }

    public void setWindSpeed(double windSpeed) {
        this.windSpeed = windSpeed;
    }

    public double getTemperature() {
        return temperature;
    }

    public void setTemperature(double temperature) {
        this.temperature = temperature;
    }

    public double getSolarIrradiance() {
        return solarIrradiance;
    }

    public void setSolarIrradiance(double solarIrradiance) {
        this.solarIrradiance = solarIrradiance;
    }

    public String getWeather() {
        return weather;
    }

    public void setWeather(String weather) {
        this.weather = weather;
    }

}
