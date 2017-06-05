# Gather timestamped GPU power usage data and store it in a text file.
#!/bin/sh

time_millis() {
  date +%s%N | cut -b1-13
}

get_all_watts() {
  nvidia-smi | sed -n "s:.*\(...\)W /.*:\1:p" | awk '{print $1}' | paste -s -d, -
}

_REC_TIME=60000  # 10000 = run for 10k milliseconds, or 10 seconds
INTERVAL=.1  # Gather data every 0.1s

START=$(time_millis)
END=$(($START + $_REC_TIME))
_TIME=$(time_millis)

while [ $_TIME -lt $END ]
do
  _TIME=$(time_millis)
  echo $_TIME,$(get_all_watts)
  sleep $INTERVAL
done
