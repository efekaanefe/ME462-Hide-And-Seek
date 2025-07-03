#!/bin/bash

SESSION="cams"

# Start a new tmux session in detached mode
tmux new-session -d -s $SESSION "./connect.sh 0"

# Split horizontally (below)
tmux split-window -v -t $SESSION "./connect.sh 1"

# Split the bottom pane again (below), to make 3 stacked panes
tmux select-pane -t $SESSION:0.1
tmux split-window -v -t $SESSION "./connect.sh 2"

# Optional: even out pane sizes
tmux select-layout -t $SESSION tiled

# Attach to the session
tmux attach-session -t $SESSION
