#!/usr/bin/env bash


checkpoint_list=(
  "/home/mprabhud/sp/DALLE-pytorch/multirun/2024-05-11/00-48-38/0/checkpoints/giddy-yogurt-37/model.pt"
  "/home/mprabhud/sp/DALLE-pytorch/multirun/2024-05-11/00-48-38/1/checkpoints/silver-meadow-40/model.pt"
  "/home/mprabhud/sp/DALLE-pytorch/multirun/2024-05-11/00-48-38/2/checkpoints/northern-sunset-37/model.pt"
  "/home/mprabhud/sp/DALLE-pytorch/multirun/2024-05-11/00-48-38/3/checkpoints/volcanic-voice-40/model.pt"
  "/home/mprabhud/sp/DALLE-pytorch/multirun/2024-05-11/00-48-38/4/checkpoints/revived-cherry-38/model.pt"
  "/home/mprabhud/sp/DALLE-pytorch/multirun/2024-05-11/00-48-38/5/checkpoints/exalted-glitter-42/model.pt"
  "/home/mprabhud/sp/DALLE-pytorch/checkpoints/scarlet-totem-47/model.pt"
  "/home/mprabhud/sp/DALLE-pytorch/checkpoints/resilient-pine-47/model.pt"
)

# resilient-pine-47: ff
# scarlet-totem-47: ff
# exalted-glitter-42: ff
# volcanic-voice-40: ro
# northern-sunset-37: ro
# giddy-yogurt-37: ro
# revived-cherry-38: ff
# silver-meadow-40: ro


# make a dictionary like above
declare -A checkpoint_dict
checkpoint_dict["resilient-pine-47"]="ff"
checkpoint_dict["scarlet-totem-47"]="ff"
checkpoint_dict["exalted-glitter-42"]="ff"
checkpoint_dict["volcanic-voice-40"]="ro"
checkpoint_dict["northern-sunset-37"]="ro"
checkpoint_dict["giddy-yogurt-37"]="ro"
checkpoint_dict["revived-cherry-38"]="ff"
checkpoint_dict["silver-meadow-40"]="ro"




for checkpoint in "${checkpoint_list[@]}"; do
    # get the checkpoint name by splitting the path and getting the second last element
    checkpoint_name=$(echo $checkpoint | rev | cut -d'/' -f2 | rev)
    # get mode from the dictionary
    mode=${checkpoint_dict[$checkpoint_name]}
    echo "Checkpoint: $checkpoint | Mode: $mode"
    python eval.py eval=eval dalle_path="$checkpoint" exp="$mode"
done