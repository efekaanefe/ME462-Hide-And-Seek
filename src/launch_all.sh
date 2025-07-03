#!/bin/bash

SESSION="multicam"

# Kill session if it exists
tmux has-session -t $SESSION 2>/dev/null
if [ $? -eq 0 ]; then
    echo "Killing existing session: $SESSION"
    tmux kill-session -t $SESSION
fi

# Start new session
tmux new-session -d -s $SESSION "bash -c './connect.sh 0; exec bash'"

# Add other camera SSH panes
tmux split-window -v -t $SESSION:0.0 "bash -c './connect.sh 1; exec bash'"
tmux split-window -v -t $SESSION:0.1 "bash -c './connect.sh 2; exec bash'"

# Go to top and split horizontally for Python scripts
tmux select-pane -t $SESSION:0.0
tmux split-window -h -t $SESSION:0.0 "bash -c 'python3 main_tcp.py --room --cam 0; exec bash'"

tmux select-pane -t $SESSION:0.3
tmux split-window -v -t $SESSION:0.3 "bash -c 'python3 main_tcp.py --room --cam 1; exec bash'"

tmux select-pane -t $SESSION:0.4
tmux split-window -h -t $SESSION:0.4 "bash -c 'python3 main_tcp_top.py; exec bash'"

tmux select-pane -t $SESSION:0.5
tmux split-window -v -t $SESSION:0.5 "bash -c 'python3 main_manager.py; exec bash'"

# Balance the layout
tmux select-layout -t $SESSION tiled

# Attach to the session
tmux attach-session -t $SESSION
