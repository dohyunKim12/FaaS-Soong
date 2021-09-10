#! /bin/bash

sudo systemctl stop kubelet

sudo docker stop $(docker ps -a -q)

sudo docker rm $(docker ps -a -q)


sudo rm -rf /var/lib/etcd
sudo rm -rf /etc/kubernetes

umount /var/lib/kubelet/pods/*/*/*/*
sudo rm -rf /var/lib/kubelet


