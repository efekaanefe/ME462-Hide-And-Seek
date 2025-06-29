## TODOs
- [x] ip of rpi4 changes over night, any way to fix it??
    sudo nano /etc/dhcpcd.conf
    interface wlan0
    static ip_address=192.168.1.101/24
    static routers=192.168.0.1
    static domain_name_servers=8.8.8.8

- [ ] try to decrease latency using things in: https://www.youtube.com/watch?v=rxtcyxV32nc&t=358s
- [x] add tracker manager script
- [x] make other rpi4s work 
