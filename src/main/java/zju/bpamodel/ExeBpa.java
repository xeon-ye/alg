package zju.bpamodel;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;

public class ExeBpa {

    public static void openPSDEdit(String PSDEditPath) {
        try{
            Runtime runtime = Runtime.getRuntime();
            Process process = runtime.exec("cmd /c start " + PSDEditPath);
            Thread.sleep(200);
            BufferedReader br = new BufferedReader(new InputStreamReader(process.getInputStream(), StandardCharsets.UTF_8));
            process = runtime.exec("cmd /c start " + PSDEditPath);
            Thread.sleep(200);
        } catch(Exception e) {
            System.out.println(e);
        }
    }

    public static void closePSDEdit() {
        try{
            Runtime runtime = Runtime.getRuntime();
            Process process = runtime.exec("cmd /c taskkill /f /im PSDEdit.exe");
            Thread.sleep(200);
            BufferedReader br = new BufferedReader(new InputStreamReader(process.getInputStream(), StandardCharsets.UTF_8));
            String line;
            while ((line = br.readLine()) != null) {
                System.out.println(line);
            }
            process = runtime.exec("cmd /c taskkill /f /im PSDEdit.exe");
            Thread.sleep(200);
            br = new BufferedReader(new InputStreamReader(process.getInputStream(), StandardCharsets.UTF_8));
            while ((line = br.readLine()) != null) {
                System.out.println(line);
            }
        } catch(Exception e) {
            System.out.println(e);
        }
    }

    public static void exePf(String pfntPath, String pfFilePath) {
        try{
            Runtime runtime = Runtime.getRuntime();
            Process process = runtime.exec("cmd /c " + pfntPath + " " + pfFilePath);
            Thread.sleep(2000);
            BufferedReader br = new BufferedReader(new InputStreamReader(process.getInputStream(), StandardCharsets.UTF_8));
            String line;
            while ((line = br.readLine()) != null) {
                System.out.println(line);
            }
        } catch(Exception e) {
            System.out.println(e);
        }
    }

    public static void exeSw(String swntPath, String bseFilePath, String swiFilePath) {
        try{
            Runtime runtime = Runtime.getRuntime();
            Process process = runtime.exec("cmd /c " + swntPath + " " + bseFilePath + " " + swiFilePath);
            Thread.sleep(5000);
            BufferedReader br = new BufferedReader(new InputStreamReader(process.getInputStream(), StandardCharsets.UTF_8));
            String line;
            while ((line = br.readLine()) != null) {
                System.out.println(line);
            }
        } catch(Exception e) {
            System.out.println(e);
        }
    }
}
