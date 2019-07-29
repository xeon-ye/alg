package zju.hems;

import java.util.List;

public class Microgrid {

    List<User> users;

    public Microgrid(List<User> users) {
        this.users = users;
    }

    public List<User> getUsers() {
        return users;
    }

    public void setUsers(List<User> users) {
        this.users = users;
    }
}
