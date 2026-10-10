#!/bin/sh
# Run on the server with root networking authority. Fixed topology only.
# DOCKER-USER sees forwarded container traffic after Docker's DNAT.
set -eu
# Docker's bridge switching must enter the IPv4 filter table. Otherwise a
# neighbour proxy on the same bridge bypasses DOCKER-USER entirely.
modprobe br_netfilter
sysctl -w net.bridge.bridge-nf-call-iptables=1 >/dev/null
for chain in DOCKER-USER REPORTING-EGRESS REPORTING-GW REPORTING-API REPORTING-PATIENT REPORTING-UI REPORTING-WORKER REPORTING-HOST; do
  iptables -w 10 -N "$chain" 2>/dev/null || true
done
# One filter-table transaction changes only our chains. Existing protection
# remains active until COMMIT; Docker and other host firewall rules are kept.
iptables-restore -w 10 --noflush <<'RULES'
*filter
-F REPORTING-EGRESS
-F REPORTING-GW
-F REPORTING-API
-F REPORTING-PATIENT
-F REPORTING-UI
-F REPORTING-WORKER
-F REPORTING-HOST
-A REPORTING-GW -m conntrack --ctstate ESTABLISHED,RELATED --ctdir REPLY -j RETURN
-A REPORTING-API -m conntrack --ctstate ESTABLISHED,RELATED --ctdir REPLY -j RETURN
-A REPORTING-PATIENT -m conntrack --ctstate ESTABLISHED,RELATED --ctdir REPLY -j RETURN
-A REPORTING-UI -m conntrack --ctstate ESTABLISHED,RELATED --ctdir REPLY -j RETURN
-A REPORTING-WORKER -m conntrack --ctstate ESTABLISHED,RELATED --ctdir REPLY -j RETURN
# Only configured proxies and necessary local services are reachable.
-A REPORTING-GW -p tcp -d 172.18.0.240 --dport 3128 -j RETURN
-A REPORTING-GW -p tcp -d 172.18.0.12 --dport 11434 -j RETURN
-A REPORTING-GW -p tcp -d 172.18.0.11 --dport 18080 -j RETURN
-A REPORTING-GW -p tcp -d 172.22.0.2 --dport 2222 -j RETURN
-A REPORTING-API -p tcp -d 172.18.0.240 --dport 3129 -j RETURN
-A REPORTING-API -p tcp -d 172.18.0.12 --dport 11434 -j RETURN
-A REPORTING-API -p tcp -d 172.18.0.4 --dport 7700 -j RETURN
-A REPORTING-API -p tcp -d 172.18.0.6 --dport 8000 -j RETURN
-A REPORTING-API -p tcp -d 172.18.0.13 --dport 8000 -j RETURN
-A REPORTING-API -p tcp -d 172.16.0.16 -m multiport --dports 80,443 -j RETURN
-A REPORTING-API -p tcp -d 172.16.0.192 -m multiport --dports 80,443 -j RETURN
-A REPORTING-PATIENT -p tcp -d 172.18.0.240 --dport 3130 -j RETURN
-A REPORTING-PATIENT -p tcp -d 172.18.0.240 --dport 3129 -j RETURN
-A REPORTING-PATIENT -p tcp -d 172.18.0.7 --dport 8010 -j RETURN
-A REPORTING-PATIENT -p tcp -d 172.19.0.2 --dport 5432 -j RETURN
-A REPORTING-PATIENT -p tcp -d 172.19.0.3 --dport 6379 -j RETURN
-A REPORTING-PATIENT -p tcp -d 172.18.0.9 --dport 6379 -j RETURN
-A REPORTING-PATIENT -p tcp -d 172.18.0.3 --dport 8000 -j RETURN
-A REPORTING-PATIENT -p tcp -d 172.19.0.4 --dport 8000 -j RETURN
-A REPORTING-UI -p tcp -d 172.18.0.240 --dport 3129 -j RETURN
-A REPORTING-UI -p tcp -d 172.18.0.12 --dport 11434 -j RETURN
-A REPORTING-UI -p tcp -d 172.18.0.4 --dport 7700 -j RETURN
-A REPORTING-UI -p tcp -d 172.18.0.6 --dport 8000 -j RETURN
-A REPORTING-UI -p tcp -d 172.18.0.13 --dport 8000 -j RETURN
-A REPORTING-UI -p tcp -d 172.18.0.11 --dport 18080 -j RETURN
-A REPORTING-UI -p tcp -d 172.18.0.7 --dport 8010 -j RETURN
-A REPORTING-UI -p tcp -d 172.16.0.16 -m multiport --dports 80,443 -j RETURN
-A REPORTING-UI -p tcp -d 172.16.0.192 -m multiport --dports 80,443 -j RETURN
-A REPORTING-WORKER -p tcp -d 172.19.0.2 --dport 5432 -j RETURN
-A REPORTING-WORKER -p tcp -d 172.18.0.12 --dport 11434 -j RETURN
-A REPORTING-GW -j REJECT
-A REPORTING-API -j REJECT
-A REPORTING-PATIENT -j REJECT
-A REPORTING-UI -j REJECT
-A REPORTING-WORKER -j REJECT
-A REPORTING-EGRESS -s 172.18.0.8 -j REPORTING-GW
-A REPORTING-EGRESS -s 172.20.0.2 -j REPORTING-GW
-A REPORTING-EGRESS -s 172.22.0.3 -j REPORTING-GW
-A REPORTING-EGRESS -s 172.18.0.7 -j REPORTING-API
-A REPORTING-EGRESS -s 172.18.0.10 -j REPORTING-PATIENT
-A REPORTING-EGRESS -s 172.19.0.5 -j REPORTING-PATIENT
-A REPORTING-EGRESS -s 172.18.0.3 -j REPORTING-PATIENT
-A REPORTING-EGRESS -s 172.19.0.4 -j REPORTING-PATIENT
-A REPORTING-EGRESS -s 172.18.0.17 -j REPORTING-WORKER
-A REPORTING-EGRESS -s 172.19.0.6 -j REPORTING-WORKER
-A REPORTING-EGRESS -s 172.18.0.14 -j REPORTING-UI
-A REPORTING-EGRESS -j RETURN
-A REPORTING-HOST -m conntrack --ctstate ESTABLISHED,RELATED --ctdir REPLY -j RETURN
-A REPORTING-HOST -s 172.18.0.8 -j REJECT
-A REPORTING-HOST -s 172.20.0.2 -j REJECT
-A REPORTING-HOST -s 172.22.0.3 -j REJECT
-A REPORTING-HOST -s 172.18.0.7 -j REJECT
-A REPORTING-HOST -s 172.18.0.10 -j REJECT
-A REPORTING-HOST -s 172.19.0.5 -j REJECT
-A REPORTING-HOST -s 172.18.0.3 -j REJECT
-A REPORTING-HOST -s 172.19.0.4 -j REJECT
-A REPORTING-HOST -s 172.18.0.17 -j REJECT
-A REPORTING-HOST -s 172.19.0.6 -j REJECT
-A REPORTING-HOST -s 172.18.0.14 -j REJECT
-A REPORTING-HOST -j RETURN
COMMIT
RULES
iptables -w 10 -C DOCKER-USER -j REPORTING-EGRESS 2>/dev/null || iptables -w 10 -I DOCKER-USER 1 -j REPORTING-EGRESS
iptables -w 10 -C INPUT -j REPORTING-HOST 2>/dev/null || iptables -w 10 -I INPUT 1 -j REPORTING-HOST
