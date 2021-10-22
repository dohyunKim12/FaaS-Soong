## NVIDA DCGM Exporter + 1 Node Exporter for Prometheus Custom ver.

**Title** | `query`

### DCGM
---
**GPU Power usage**  | `DCGM_FI_DEV_POWER_USAGE{instance=~"${instance}", gpu=~"${gpu}"}` <br>

![image](https://user-images.githubusercontent.com/72643027/138420273-e4b07ffa-e93a-4374-b9d3-831e1798eec3.png)

**GPU Temperature** | `DCGM_FI_DEV_GPU_TEMP{instance=~"${instance}", gpu=~"${gpu}"}` <br>

![image](https://user-images.githubusercontent.com/72643027/138420453-73364795-18d4-48a3-89eb-fb31294afe6f.png)

**GPU Power Total** | `sum(DCGM_FI_DEV_POWER_USAGE{instance=~"${instance}", gpu=~"${gpu}"})` <br>

![image](https://user-images.githubusercontent.com/72643027/138420609-2777c4b6-24f9-4ca9-862b-db14c098243c.png)


### 1 Node Exporter for Prometheus
---
**Server Resource Overview** | `node_uname_info{origin_prometheus=~"$origin_prometheus",job=~"$job"} - 0` <br>
`sum(time() - node_boot_time_seconds{origin_prometheus=~"$origin_prometheus",job=~"$job"})by(instance)` <br>
`node_memory_MemTotal_bytes{origin_prometheus=~"$origin_prometheus",job=~"$job"} - 0` <br>
