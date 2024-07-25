# parse and visualize the logfile
import numpy as np
import matplotlib.pyplot as plt

sz = "124M"

loss_baseline = {
    "124M": 3.2924,
}[sz]
hella2_baseline = { # HellaSwag for GPT-2
    "124M": 0.294463,
    "350M": 0.375224,
    "774M": 0.431986,
    "1558M": 0.488946,
}[sz]
hella3_baseline = { # HellaSwag for GPT-3
    "124M": 0.337,
    "350M": 0.436,
    "774M": 0.510,
    "1558M": 0.547,
}[sz]

# load the log file
# define a list of log files
log_files = ["log/log.txt"]

# initialize a dictionary to store the data from each log file
streams_dict = {}

# loop over the log files
for log_file in log_files:
    # load the log file
    with open(log_file, "r") as f:
        lines = f.readlines()

    # parse the individual lines, group by stream (train,val,hella)
    streams = {}
    for line in lines:
        step, stream, val = line.strip().split()
        if stream not in streams:
            streams[stream] = {}
        streams[stream][int(step)] = float(val)

    # convert each stream from {step: val} to (steps[], vals[])
    # so it's easier for plotting
    streams_xy = {}
    for k, v in streams.items():
        # get all (step, val) items, sort them
        xy = sorted(list(v.items()))
        # unpack the list of tuples to tuple of lists
        streams_xy[k] = list(zip(*xy))

    # store the data in the dictionary
    streams_dict[log_file] = streams_xy

# create figure
plt.figure(figsize=(16, 6))

# Panel 1: losses: both train and val
plt.subplot(121)
for log_file, streams_xy in streams_dict.items():
    xs, ys = streams_xy["train"] # training loss
    plt.plot(xs, ys, label=f'{log_file} train loss')
    xs, ys = streams_xy["val"] # validation loss
    plt.plot(xs, ys, label=f'{log_file} val loss')
# horizontal line at GPT-2 baseline
if loss_baseline is not None:
    plt.axhline(y=loss_baseline, color='r', linestyle='--', label=f"OpenAI GPT-2 ({sz}) checkpoint val loss")
plt.xlabel("steps")
plt.ylabel("loss")
plt.yscale('log')
plt.ylim(top=4.0)
plt.legend()
plt.title("Loss")

# Panel 2: HellaSwag eval
plt.subplot(122)
for log_file, streams_xy in streams_dict.items():
    xs, ys = streams_xy["hella"] # HellaSwag eval
    plt.plot(xs, ys, label=f"{log_file}")
# horizontal line at GPT-2 baseline
if hella2_baseline:
    plt.axhline(y=hella2_baseline, color='r', linestyle='--', label=f"OpenAI GPT-2 ({sz}) checkpoint")
if hella3_baseline:
    plt.axhline(y=hella3_baseline, color='g', linestyle='--', label=f"OpenAI GPT-3 ({sz}) checkpoint")
plt.xlabel("steps")
plt.ylabel("accuracy")
plt.legend()
plt.title("HellaSwag eval")