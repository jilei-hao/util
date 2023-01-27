#! /bin/bash

# validate inputs
if ! test -d $1; then
  echo "Input dir 1 does not exist: $1"
  exit -1
fi

if ! test -d $2; then
  echo "Input dir 2 does not exist: $2"
  exit -1
fi

if ! test -d $3; then 
  echo "Working dir does not exist, creating one: $3"
  mkdir $3
fi

# setting variables
dir_1=$1
dir_2=$2
dir_work=$3

# work directory for each tracer (thresholded images, etc.)
dw1=$dir_work/tracer_1
dw2=$dir_work/tracer_2

if ! test -d $dw1; then
  mkdir $dw1;
fi

if ! test -d $dw2; then
  mkdir $dw2;
fi

for f1 in $dir_1/*; do
  fn=${f1##*/} # extract filename from the path
  fnonly=${fn%.*.*}
  echo ""
  echo ""
  echo "==================================================================="
  echo "-- PROCESSING FILE: $fnonly "
  echo "==================================================================="
  echo ""
  
  if ! test -f $dir_2/$fn; then # check file existence in dir_2
    echo "-- $dir_2/$fn does not exist! Will skip the file."
    continue
  fi

  fn1=$dir_1/$fn
  fn2=$dir_2/$fn

  # work directory for each seg file
  
  dws1=$dw1/$fnonly
  dws2=$dw2/$fnonly

  if ! test -d $dws1; then
    mkdir $dws1;
  fi

  if ! test -d $dws2; then
    mkdir $dws2;
  fi

  # parse number of labels
  info=$(c3d $fn1 -info)
  search_str="range = ["
  echo "info=${info}"
  rest=${info#*$search_str}
  r1=$(echo $rest | awk -F'[^0-9]+' '{ print $1 }') # lower bound
  r2=$(echo $rest | awk -F'[^0-9]+' '{ print $2 }') # upper bound
  nL=$(($r2 - $r1))
  echo "Number of Labels: ${nL}"

  # whole structure comparison
  # -- binarizing fn1
  fn1_binary_all="$dws1/binary_all_labels.nii.gz"
  cmd="c3d -background 0 $fn1 -binarize -o ${fn1_binary_all}"
  $cmd

  # -- vtkleveset fn1
  fn1_mesh_all="$dws1/mesh_all_labels.vtk"
  cmd="vtklevelset $fn1_binary_all $fn1_mesh_all 1"
  $cmd

  # -- binarizing fn2
  fn2_binary_all="$dws2/binary_all_labels.nii.gz"
  cmd="c3d -background 0 $fn2 -binarize -o ${fn2_binary_all}"
  $cmd

  # -- vtkleveset fn2
  fn2_mesh_all="$dws2/mesh_all_labels.vtk"
  cmd="vtklevelset $fn2_binary_all $fn2_mesh_all 1"
  $cmd

  # -- computing DICE
  echo ""
  echo "==================================================="
  echo "-- ALL LABEL RESULT"
  echo "==================================================="
  echo "-- DICE ---------------------"
  cmd="c3d -verbose ${fn1_binary_all} ${fn2_binary_all} -overlap 1"
  $cmd
  echo ""
  echo "-- MESH DIFF ----------------"
  cmd="meshdiff ${fn1_mesh_all} ${fn2_mesh_all}"
  $cmd
  echo "==================================================="
  echo ""
  echo ""



  echo ""
  echo "==================================================="
  echo "-- LABEL-WISE COMPARISON"
  echo "==================================================="
  echo ""

  # label-wise comparison
  for (( lb=1; lb<=$nL; lb++)); do
    echo ""
    echo "=================================="
    echo "-- PROCESSING LABEL: ${lb}"
    echo "=================================="
    echo ""
    
    # binarizing fn1
    fn1_binary="$dws1/binary_label_${lb}.nii.gz"
    cmd="c3d $fn1 -threshold ${lb} ${lb} 1 0 -o $fn1_binary"
    #echo $cmd
    $cmd

    # vtkleveset fn1
    fn1_mesh="$dws1/mesh_label_${lb}.vtk"
    cmd="vtklevelset $fn1_binary $fn1_mesh 1"
    $cmd

    # binarizing fn2
    fn2_binary="$dws2/binary_label_${lb}.nii.gz"
    cmd="c3d $fn2 -threshold ${lb} ${lb} 1 0 -o $fn2_binary"
    #echo $cmd
    $cmd

    # vtkleveset fn2
    fn2_mesh="$dws2/mesh_label_${lb}.vtk"
    cmd="vtklevelset $fn2_binary $fn2_mesh 1"
    $cmd

    # computing dice
    echo ""
    echo "==================================================="
    echo "-- LABEL ${lb} RESULT"
    echo "==================================================="
    echo "-- DICE ---------------------"
    cmd="c3d -verbose ${fn1_binary} ${fn2_binary} -overlap 1"
    $cmd
    echo ""
    echo "-- MESH DIFF ----------------"
    cmd="meshdiff ${fn1_mesh} ${fn2_mesh}"
    $cmd
    echo "=================================================="
    echo ""
    echo ""

  done

done