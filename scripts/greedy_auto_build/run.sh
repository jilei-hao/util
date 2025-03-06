#! /bin/bash

# get script directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
echo $DIR

## ITK BUILD

# clone itk
if [ ! -d "$DIR/itk" ]; then
  git clone https://github.com/InsightSoftwareConsortium/ITK.git $DIR/itk
fi

cd $DIR/itk
git checkout v5.4.0

# configure
cmake \
-G "Unix Makefiles" \
-S $DIR/itk \
-B $DIR/itk-build \
-D CMAKE_BUILD_TYPE=Release \
-D BUILD_SHARED_LIBS=OFF \
-D BUILD_TESTING=OFF \
-D BUILD_EXAMPLES=OFF \
-D Module_MorphologicalContourInterpolation=ON

# build
cmake --build $DIR/itk-build -j 8



## VTK BUILD

# clone vtk
if [ ! -d "$DIR/vtk" ]; then
  git clone https://gitlab.kitware.com/vtk/vtk.git $DIR/vtk
fi

cd $DIR/vtk
git checkout v9.3.0

# configure
cmake \
-G "Unix Makefiles" \
-S $DIR/vtk \
-B $DIR/vtk-build \
-D CMAKE_BUILD_TYPE=Release \
-D BUILD_SHARED_LIBS=OFF \
-D VTK_BUILD_TESTING=OFF \
-D VTK_BUILD_EXAMPLES=OFF

# build
cmake --build $DIR/vtk-build -j 8


## BUILD GREEDY

# clone greedy
if [ ! -d "$DIR/greedy" ]; then
  git clone https://github.com/pyushkevich/greedy $DIR/greedy
fi

# configure
cmake \
-G "Unix Makefiles" \
-S $DIR/greedy \
-B $DIR/greedy-build \
-D CMAKE_BUILD_TYPE=Release \
-D BUILD_SHARED_LIBS=OFF \
-D ITK_DIR=$DIR/itk-build \
-D VTK_DIR=$DIR/vtk-build

# build
cmake --build $DIR/greedy-build -j 8






