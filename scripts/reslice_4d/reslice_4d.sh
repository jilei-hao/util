# #! /bin/bash

# # usage: reslice_4d.sh <fixed_image> <moving_image> <reslice_command> <working_directory>
# # example: reslice_4d.sh fix.nii.gz mov.nii.gz -reslice-identity /work
# # example: reslice_4d.sh fix.nii.gz mov.nii.gz -reslice-itk /path/to/transform.txt /work

imgFix=$1
imgMov=$2
resliceCmd=$3
dirWork=$4

fnMov3d="mov3d"
fnFix3d="fix3d"

echo "-- Fix Image: $imgFix"
echo "-- Mov Image: $imgMov"
echo "-- Reslice Command: $resliceCmd"
echo "-- Working Directory: $dirWork"

echo "-- Extracting fix 3d volumes"
c4d $imgFix -slice w 0:-1 -oo ${dirWork}/${fnFix3d}_%02d.nii.gz

echo "-- Extracting mov 3d volumes"
c4d $imgMov -slice w 0:-1 -oo ${dirWork}/${fnMov3d}_%02d.nii.gz


echo "-- Reslicing 3d volumes"
for i in $(ls ${dirWork}/${fnMov3d}_*.nii.gz); do
    idx=$(echo $i | awk -F_ '{print $NF}' | awk -F. '{print $1}')
    echo "-- Reslicing volume: $idx"
    fnOut=${dirWork}/mov3d-resliced_${idx}.nii.gz
    cmd="c3d -int 0 ${dirWork}/${fnFix3d}_${idx}.nii.gz ${dirWork}/${fnMov3d}_${idx}.nii.gz \
        ${resliceCmd} -o ${fnOut}"

    # echo $cmd
    eval $cmd
done

echo "-- Tiling resliced volumes into 4d"
c4d ${dirWork}/mov3d-resliced_*.nii.gz -tile w -o ${dirWork}/mov4d_resliced.nii.gz

